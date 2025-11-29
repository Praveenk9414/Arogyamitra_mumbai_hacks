# tools/validate_procedure.py
"""
Procedure validator: accepts a procedure name or a report path/text.
Uses small guidelines for quick deterministic answers; otherwise uses local LLM
to produce structured Necessity / Rationale / Alternatives.
"""

import os, re
from typing import Dict, Any
from llm import safe_llm_summary
from utils.helpers import normalize_proc_name

# simple guideline dict (expand later or load from data/guidelines.json)
GUIDELINES = {
    "appendectomy": {
        "procedure": "Appendectomy",
        "indications": [
            "Appendicitis with imaging evidence (enlarged appendix, wall thickening, fat stranding)",
            "Perforated appendix",
            "Recurrent appendicitis"
        ],
        "conservative_alternatives": [
            "Antibiotic therapy trial if early and clinically stable",
            "Observation with repeat imaging"
        ],
        "notes": "Surgery generally recommended if imaging + clinical signs strongly support appendicitis."
    },
    "partial_medial_meniscus_tear": {
        "procedure": "Partial medial meniscus tear",
        "indications": [
            "Mechanical locking or persistent pain after 6-8 weeks conservative therapy",
            "Large displaced tear on MRI in young active patients"
        ],
        "conservative_alternatives": [
            "Physiotherapy",
            "NSAIDs and activity modification",
            "Steroid injection"
        ],
        "notes": "Partial tears often respond to conservative care before surgery."
    }
}

LLM_PROMPT_TEMPLATE = """
You are a medical procedure validator. Given this input (either a short report excerpt or a doctor's recommendation),
decide whether the recommended surgery is: Required / Possibly avoidable / Not recommended.
Return EXACTLY these labeled fields (no other text):

Necessity: <Required | Possibly avoidable | Not recommended>
Rationale: <one or two sentence rationale>
Alternatives:
- <alternative 1>
- <alternative 2>

Input:
{input_text}
"""

def _read_file_if_path(input_text_or_path: str) -> str:
    """Return text: if argument is existing file path -> extract its contents; else return string itself."""
    if os.path.exists(input_text_or_path):
        # use translate tool's extraction logic to keep simple (avoid circular import)
        ext = os.path.splitext(input_text_or_path)[1].lower()
        if ext == ".pdf":
            try:
                import fitz
                doc = fitz.open(input_text_or_path)
                pages = [p.get_text("text") for p in doc]
                doc.close()
                return "\n\n".join(pages)
            except Exception:
                with open(input_text_or_path, "r", encoding="utf-8", errors="ignore") as f:
                    return f.read()
        else:
            with open(input_text_or_path, "r", encoding="utf-8", errors="ignore") as f:
                return f.read()
    return input_text_or_path

def validate_procedure_tool(input_text_or_path: str, model: str = None) -> Dict[str, Any]:
    """
    If a guideline match exists, return a quick structured result.
    Otherwise call local LLM to produce a structured response.
    """
    if not input_text_or_path or not input_text_or_path.strip():
        return {"status":"error","error":"No input provided."}

    text = _read_file_if_path(input_text_or_path)

    # try to detect a procedure short name in the text
    proc_guess = normalize_proc_name(text)
    guideline = GUIDELINES.get(proc_guess)
    if guideline:
        necessity = "Possibly avoidable"
        if re.search(r"(perforat|fat stranding|wall thickening|diameter\s*\d+\.?\d*\s*mm|free fluid|perforation)", text, re.I):
            necessity = "Required"
        return {
            "status":"ok",
            "procedure":guideline["procedure"],
            "necessity":necessity,
            "rationale":guideline.get("notes",""),
            "alternatives":guideline.get("conservative_alternatives",[])
        }

    # Fallback: use LLM to reason
    prompt = LLM_PROMPT_TEMPLATE.format(input_text=text)
    try:
        llm_out = safe_llm_summary(prompt, model=model)
    except Exception as e:
        return {"status":"error","error":f"LLM call failed: {e}"}

    # parse LLM output
    res = {"status":"ok", "raw_llm": llm_out}
    m1 = re.search(r"Necessity:\s*(.+)", llm_out, re.I)
    m2 = re.search(r"Rationale:\s*([\s\S]*?)\nAlternatives:", llm_out, re.I)
    m3 = re.search(r"Alternatives:\s*([\s\S]+)", llm_out, re.I)
    res["necessity"] = m1.group(1).strip() if m1 else "Unknown"
    res["rationale"] = (m2.group(1).strip() if m2 else "").strip()
    if m3:
        lines = [l.strip("- ").strip() for l in m3.group(1).splitlines() if l.strip()]
        res["alternatives"] = lines
    else:
        res["alternatives"] = []
    return res
