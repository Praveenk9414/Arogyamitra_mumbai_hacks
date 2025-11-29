"""
agent.py

Agent that aggregates insights from:
 - procedure validator (tools.validate_procedure.validate_procedure_tool)
 - bill auditor (tools.audit_bill.audit_bill)
 - (optional) translator (simple translator function / LLM on reports)
 - RAG index (tools.rag_interface) for retrieval from ingested docs

Usage (programmatic):
  from agent import Agent
  a = Agent(session_id="test01")
  a.ingest_files(["patient_1_report.pdf", "bill_1.pdf"])
  summary = a.build_aggregate_summary()
  resp = a.ask("Is appendectomy necessary for this patient?", use_rag=True, use_llm=True)
  print(resp["answer"])

Also provides a simple CLI interactive mode when run as __main__.
"""

import os
import json
import hashlib
import tempfile
import traceback
from typing import List, Dict, Any, Optional

# Local LLM wrapper
try:
    from llm import safe_llm_summary
    LLM_AVAILABLE = True
except Exception:
    safe_llm_summary = None
    LLM_AVAILABLE = False

# Tools (may raise ImportError if not present)
try:
    from tools.validate_procedure import validate_procedure_tool, interactive_question_on_report
except Exception:
    validate_procedure_tool = None
    interactive_question_on_report = None

try:
    from tools.audit_bill import audit_bill
except Exception:
    audit_bill = None

try:
    from extractors.text_extractor import extract_text_and_tables, chunk_documents
except Exception:
    extract_text_and_tables = None
    chunk_documents = None

try:
    from tools import rag_interface
except Exception:
    rag_interface = None

# config
RAG_BASE = "agent_rag_sessions"
os.makedirs(RAG_BASE, exist_ok=True)


class Agent:
    """
    Aggregation agent.

    - session_id: optional id (if using RAG we create a session collection under RAG_BASE/session_id)
    - store: small in-memory store of ingested file metadata + summaries
    """

    def __init__(self, session_id: Optional[str] = None, embedding_model: str = "nomic-embed-text:latest", llama_model: Optional[str] = None):
        self.session_id = session_id or hashlib.sha1(os.urandom(8)).hexdigest()[:8]
        self.embedding_model = embedding_model
        self.llama_model = llama_model  # passed into rag_interface.call_ollama_single where appropriate
        self.files: List[Dict[str, Any]] = []  # each: {"path", "type", "summary", "raw_text" (optional)}
        self.structured_insights: Dict[str, Any] = {}  # aggregated outputs
        # Chroma collection handle (lazy)
        self._collection = None

    # ---------------- ingestion ----------------
    def ingest_files(self, file_paths: List[str], types: Optional[List[str]] = None, ingest_to_rag: bool = True) -> Dict[str, Any]:
        """
        Ingest files. file_paths: list of filesystem paths (pdf/txt).
        types: optional list the same length specifying 'report'|'bill'|'other'. If None, we heuristically determine.
        ingest_to_rag: whether to chunk + upsert to chroma collection for retrieval.
        Returns dict with per-file results.
        """
        results = {}
        for i, path in enumerate(file_paths):
            try:
                ftype = None
                if types and i < len(types):
                    ftype = types[i]
                else:
                    # heuristics
                    ln = os.path.basename(path).lower()
                    if "bill" in ln or "invoice" in ln:
                        ftype = "bill"
                    elif "report" in ln or path.lower().endswith(".pdf") or "ct" in ln or "xray" in ln or "lab" in ln:
                        ftype = "report"
                    else:
                        ftype = "other"

                # extract short raw text for small summary generation (use validate_procedure extraction)
                text = None
                if os.path.exists(path):
                    # try to extract via tools.validate_procedure internal extractor if present
                    if validate_procedure_tool is not None:
                        # call validator's internal extractor by calling validate with raw text? safer: call interactive_question_on_report's extractor
                        try:
                            # interactive_question_on_report internally extracts; call it with use_llm=False to get context preview
                            if interactive_question_on_report:
                                preview = interactive_question_on_report(path, "summarize for ingestion", use_llm=False)
                                # the function returns a dict; preview["context"] may exist
                                if isinstance(preview, dict) and preview.get("context"):
                                    text = preview["context"].get("report_excerpt") if isinstance(preview["context"], dict) else None
                        except Exception:
                            text = None
                    # fallback: read file as bytes/text or use extractors.text_extractor
                    if not text and extract_text_and_tables:
                        try:
                            docs = extract_text_and_tables(path)
                            # join first few docs as raw_text
                            text = "\n\n".join([d.page_content for d in docs[:4]])
                        except Exception:
                            text = None
                    if not text:
                        # last resort: open file and try to read text (works for txt)
                        try:
                            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                                text = f.read()[:4000]
                        except Exception:
                            text = None

                # run domain tools
                summary = {}
                if ftype == "report" and validate_procedure_tool:
                    try:
                        out = validate_procedure_tool(path, procedure_hint=None, use_llm=False)
                        summary["validator"] = out
                    except Exception as e:
                        summary["validator_error"] = str(e)
                if ftype == "bill" and audit_bill:
                    try:
                        out = audit_bill(path, use_llm=False)  # expect structured dict
                        summary["auditor"] = out
                    except Exception as e:
                        summary["auditor_error"] = str(e)

                # store
                self.files.append({"path": path, "type": ftype, "summary": summary, "raw_text": (text or "")})
                results[path] = {"ok": True, "type": ftype, "summary_keys": list(summary.keys())}

            except Exception as e:
                results[path] = {"ok": False, "error": str(e)}

        # optional: ingest docs into RAG
        if ingest_to_rag and rag_interface and extract_text_and_tables and chunk_documents:
            try:
                coll = self._ensure_rag_collection()
                # for each file, extract+chunk and upsert
                for p in file_paths:
                    if not os.path.exists(p):
                        continue
                    docs = extract_text_and_tables(p)
                    chunks = chunk_documents(docs)
                    payload = [{"text": d.page_content, "metadata": d.metadata} for d in chunks]
                    # ids prefix by filename safe
                    ids_prefix = os.path.basename(p).replace(" ", "_")
                    rag_interface.upsert_documents_to_chroma(coll, payload, ids_prefix=ids_prefix)
            except Exception as e:
                results["_rag_ingest_error"] = str(e)

        # update aggregated structured insights
        self._aggregate_structured_insights()
        return results

    # ---------------- rag collection ----------------
    def _ensure_rag_collection(self):
        if self._collection is not None:
            return self._collection
        if rag_interface is None:
            raise RuntimeError("rag_interface not available")
        chroma_path = os.path.join(RAG_BASE, self.session_id)
        coll_name = f"agent_{self.session_id}"
        coll = rag_interface.get_chroma_collection(chroma_path, coll_name, embedding_model=self.embedding_model)
        self._collection = coll
        return coll

    # ---------------- aggregation ----------------
    def _aggregate_structured_insights(self):
        """
        Build/update self.structured_insights by collecting validator/auditor summaries from each file.
        """
        agg = {"reports": [], "bills": [], "counts": {"files": len(self.files)}}
        for f in self.files:
            summary = f.get("summary", {})
            if f["type"] == "report":
                agg["reports"].append({"path": f["path"], "validator": summary.get("validator", summary.get("validator_error"))})
            elif f["type"] == "bill":
                agg["bills"].append({"path": f["path"], "auditor": summary.get("auditor", summary.get("auditor_error"))})
            else:
                agg.setdefault("others", []).append({"path": f["path"], "summary": summary})
        self.structured_insights = agg
        return agg

    def build_aggregate_summary(self, max_chars: int = 1500) -> str:
        """
        Create a readable summary string covering:
         - number of files
         - brief lines per report: detected procedure + necessity (if available)
         - brief lines per bill: top flagged issues (if available)
        """
        parts = []
        agg = self.structured_insights or self._aggregate_structured_insights()
        parts.append(f"Files ingested: {agg.get('counts',{}).get('files', len(self.files))}")
        if agg.get("reports"):
            parts.append("\nReports summary:")
            for r in agg["reports"]:
                p = r.get("validator")
                if not p:
                    parts.append(f"- {os.path.basename(r['path'])}: no validator output")
                    continue
                proc = p.get("procedure_display") or p.get("procedure") or "unknown"
                necessity = p.get("necessity") or "unknown"
                rationale = (p.get("rationale") or "")
                parts.append(f"- {os.path.basename(r['path'])}: procedure={proc}, necessity={necessity}. {rationale[:200]}")
        if agg.get("bills"):
            parts.append("\nBills summary:")
            for b in agg["bills"]:
                a = b.get("auditor")
                if not a:
                    parts.append(f"- {os.path.basename(b['path'])}: no auditor output")
                    continue
                # auditor expected to give flags and dispute_email or parsed info
                flags = a.get("flags") if isinstance(a, dict) else None
                # fallback: try keys
                if flags:
                    parts.append(f"- {os.path.basename(b['path'])}: flagged {len(flags)} items")
                else:
                    parts.append(f"- {os.path.basename(b['path'])}: auditor output present")
        # include short raw_text previews for debugging (optional)
        if len(self.files) > 0:
            parts.append("\nSample text previews:")
            for f in self.files[:3]:
                preview = (f.get("raw_text") or "")[:300].replace("\n", " ")
                parts.append(f"- {os.path.basename(f['path'])}: {preview}")

        summary = "\n".join(parts)
        if len(summary) > max_chars:
            return summary[:max_chars] + " ..."
        return summary

    # ---------------- querying & answering ----------------
    def _build_prompt_for_question(self, question: str, agg_summary: str, retrieved_context: Optional[str]) -> str:
        """
        Build a safe prompt combining:
         - instruction: answer only using provided context
         - aggregate summary (short)
         - retrieved context (longer, optional)
         - user question
        """
        prompt_parts = [
            "You are Arogyamitra, an AI medical assistant. Answer the question using ONLY the information provided below (AGGREGATE SUMMARY and CONTEXT).",
            "If the information is insufficient to answer, say: \"I don't know based on the provided information; consult the treating physician.\"",
            "",
            "AGGREGATE SUMMARY:",
            agg_summary,
            "",
        ]
        if retrieved_context:
            prompt_parts += ["CONTEXT (retrieved documents):", retrieved_context, ""]
        else:
            prompt_parts += ["CONTEXT: None (no retrieval)"]
        prompt_parts += ["USER QUESTION:", question, "", "Please answer concisely and include one-line Suggested next step if applicable."]
        return "\n".join(prompt_parts)

    def ask(self, question: str, use_rag: bool = True, top_k: int = 8, rerank_k: int = 3, use_llm: bool = True) -> Dict[str, Any]:
        """
        Main agent query entry.
        Returns:
          {
            "ok": True/False,
            "answer": str,
            "prompt_used": str,
            "retrieved": { "docs": [...], "citations": [...] } (if rag used),
            "aggregate_summary": str,
            "error": str (if any)
          }
        """
        try:
            agg_summary = self.build_aggregate_summary()
            retrieved_text = None
            citations = []

            if use_rag and rag_interface:
                try:
                    coll = self._ensure_rag_collection()
                    res = rag_interface.retrieve_top_k(coll, question, k=top_k)
                    docs = res.get("documents", [])
                    ids = res.get("ids", [])
                    # rerank for final top-k context
                    joined, citations = rag_interface.rerank_with_crossencoder(question, docs, top_k=rerank_k)
                    retrieved_text = joined
                except Exception as e:
                    retrieved_text = None
                    # continue — we still can answer with aggregate summary
                    citations = []
            # build prompt
            prompt = self._build_prompt_for_question(question, agg_summary, retrieved_text)
            if not use_llm:
                # deterministic fallback: if question asks to validate a procedure, try to reuse validator outputs
                ql = question.lower()
                if any(w in ql for w in ("necessary", "validate", "should", "required")):
                    # naive: if any report has validator and necessity Required -> answer Required
                    for r in self.structured_insights.get("reports", []):
                        v = r.get("validator")
                        if isinstance(v, dict) and v.get("necessity") == "Required":
                            return {"ok": True, "answer": f"Required — found in {os.path.basename(r['path'])}. Suggested next step: urgent clinical review.", "prompt_used": prompt, "retrieved": None, "aggregate_summary": agg_summary}
                    return {"ok": True, "answer": "Possibly avoidable or insufficient evidence in summaries. Consult treating clinician.", "prompt_used": prompt, "retrieved": None, "aggregate_summary": agg_summary}
                return {"ok": True, "answer": "LLM disabled — enable use_llm=True for free-text answers.", "prompt_used": prompt, "retrieved": None, "aggregate_summary": agg_summary}

            # call LLM
            if not LLM_AVAILABLE or safe_llm_summary is None:
                return {"ok": False, "error": "Local LLM wrapper not available (llm.safe_llm_summary)", "aggregate_summary": agg_summary}

            llm_out = safe_llm_summary(prompt)
            # if llm_out appears like a diagnostic, surface it
            if isinstance(llm_out, str) and llm_out.startswith("LLM (Ollama) unavailable"):
                return {"ok": False, "error": llm_out, "aggregate_summary": agg_summary}

            return {
                "ok": True,
                "answer": llm_out.strip() if isinstance(llm_out, str) else str(llm_out),
                "prompt_used": prompt,
                "retrieved": {"text": retrieved_text, "citations": citations},
                "aggregate_summary": agg_summary,
            }

        except Exception as e:
            tb = traceback.format_exc()
            return {"ok": False, "error": str(e), "traceback": tb}

    # ---------------- small helpers for CLI or UI integration ----------------
    def list_ingested_files(self) -> List[str]:
        return [f["path"] for f in self.files]

    def clear(self):
        self.files = []
        self.structured_insights = {}
        self._collection = None
        # optionally clear rag session directory
        try:
            p = os.path.join(RAG_BASE, self.session_id)
            if os.path.exists(p):
                # do not delete by default
                pass
        except Exception:
            pass


# ---------------- CLI quick interactive ----------------
def _cli_loop(agent: Agent):
    print(f"Arogyamitra Agent (session {agent.session_id})")
    print("Type 'help' for commands. Example usage:")
    print("  ingest path/to/patient_1_report.pdf path/to/bill_1.pdf")
    print("  summary")
    print("  ask Is appendectomy necessary?")
    print("  exit")

    while True:
        try:
            raw = input("agent> ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nBye")
            break
        if not raw:
            continue
        cmd, *rest = raw.split(" ")
        if cmd == "help":
            print("Commands: ingest <files...> | summary | ask <question> | list | exit")
        elif cmd == "ingest":
            if not rest:
                print("Usage: ingest file1 [file2 ...]")
                continue
            paths = rest
            res = agent.ingest_files(paths, ingest_to_rag=True)
            print("Ingest results:")
            print(json.dumps(res, indent=2))
        elif cmd == "summary":
            print(agent.build_aggregate_summary())
        elif cmd == "list":
            print(agent.list_ingested_files())
        elif cmd == "ask":
            q = " ".join(rest)
            if not q:
                print("Usage: ask <your question>")
                continue
            out = agent.ask(q, use_rag=True, use_llm=True)
            if out.get("ok"):
                print("\n=== AGENT ANSWER ===\n")
                print(out["answer"])
                print("\n=== AGGREGATE SUMMARY ===\n")
                print(out["aggregate_summary"])
                if out.get("retrieved") and out["retrieved"].get("citations"):
                    print("\nCitations:", out["retrieved"]["citations"])
            else:
                print("Error:", out.get("error") or out)
        elif cmd in ("exit", "quit"):
            break
        else:
            print("Unknown command. Type 'help'.")

# ---------------- run as script ----------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--session", type=str, default=None)
    parser.add_argument("--ingest", nargs="*", help="Files to ingest at startup")
    args = parser.parse_args()

    ag = Agent(session_id=args.session)
    if args.ingest:
        print("Ingesting:", args.ingest)
        r = ag.ingest_files(args.ingest, ingest_to_rag=True)
        print("Ingested:", r)
    _cli_loop(ag)
