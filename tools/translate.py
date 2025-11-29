# tools/translate.py
"""
Translate Report tool — uses PyMuPDF to extract text from uploaded file (PDF/TXT)
and uses local LLM (Ollama via llm.safe_llm_summary) to create a structured
summary: Summary, Explanation (bullets), Next Steps, Urgency.
"""

import os
import re
from typing import Dict, Any

from llm import safe_llm_summary

# optional PyMuPDF for extracting PDFs
try:
    import fitz  # pymupdf
    MUPDF_AVAILABLE = True
except Exception:
    MUPDF_AVAILABLE = False

PROMPT_TEMPLATE = """
You are a medical translator assistant. Translate the following medical report into simple,
patient-friendly English. Output EXACTLY these labeled sections (no extra text):

Summary:
- <1-2 sentence plain summary>

Explanation:
- <bullet points of key findings>

Next Steps:
- <2-3 actionable suggestions (e.g., see specialist, get tests, emergency)>

Urgency: <URGENT or ROUTINE>

Report:
{report}
"""

def extract_text_from_pdf(path: str) -> str:
    if not MUPDF_AVAILABLE:
        raise RuntimeError("PyMuPDF not installed. Install with: pip install pymupdf")
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    doc = fitz.open(path)
    pages = []
    for page in doc:
        try:
            pages.append(page.get_text("text"))
        except Exception:
            pages.append(page.get_text())
    doc.close()
    return "\n\n".join(pages)

def extract_text_from_file(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        return extract_text_from_pdf(path)
    if ext in [".txt", ".md"]:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    # fallback: try PyMuPDF open
    if MUPDF_AVAILABLE:
        return extract_text_from_pdf(path)
    raise RuntimeError(f"Unsupported file type: {ext}")

def translate_report_input(input_text_or_path: str, model: str = None) -> Dict[str, Any]:
    """
    If input_text_or_path is an existing file path -> extract text from file.
    Otherwise treat as raw text.
    Then call local LLM to generate structured summary.
    """
    try:
        if os.path.exists(input_text_or_path):
            source = "file"
            extracted = extract_text_from_file(input_text_or_path)
        else:
            source = "text"
            extracted = str(input_text_or_path)
    except Exception as e:
        return {"status": "error", "error": str(e)}

    # build prompt and call local LLM
    prompt = PROMPT_TEMPLATE.format(report=extracted)
    try:
        llm_out = safe_llm_summary(prompt, model=model)
    except Exception as e:
        # fallback - simple heuristic summary if LLM fails
        # Very small fallback: take lines with keywords
        keywords = ["lesion","fracture","opacity","nodule","mass","appendix","appendicitis","effusion","tear","fat stranding"]
        lines = [l.strip() for l in extracted.splitlines() if l.strip()]
        findings = [l for l in lines if any(k in l.lower() for k in keywords)]
        summary = "Possible findings: " + (", ".join(findings[:2]) if findings else "No clear findings.")
        fallback_text = f"Summary:\n- {summary}\n\nExplanation:\n- " + "\n- ".join(findings[:3]) + "\n\nNext Steps:\n- Follow up with your doctor.\n\nUrgency: ROUTINE"
        return {"status": "ok", "source": source, "extracted_text": extracted, "human_text": fallback_text, "raw_llm": None}

    # Try to parse the returned labeled sections into a structured dict (robust but simple)
    struct = {"summary":"", "explanation":[], "next_steps":[], "urgency":"ROUTINE"}
    text = llm_out if isinstance(llm_out, str) else str(llm_out)
    # Summary
    m = re.search(r"Summary:\s*-?\s*([\s\S]*?)\n\s*Explanation:", text, re.I)
    if m:
        struct["summary"] = m.group(1).strip()
    # Explanation bullets
    m = re.search(r"Explanation:\s*([\s\S]*?)\n\s*Next Steps:", text, re.I)
    if m:
        bullets = re.findall(r"^-+\s*(.+)$", m.group(1), re.M)
        if not bullets:
            bullets = [l.strip("- ").strip() for l in m.group(1).splitlines() if l.strip()]
        struct["explanation"] = bullets
    # Next Steps
    m = re.search(r"Next Steps:\s*([\s\S]*?)\n\s*Urgency:", text, re.I)
    if m:
        bullets = re.findall(r"^-+\s*(.+)$", m.group(1), re.M)
        if not bullets:
            bullets = [l.strip("- ").strip() for l in m.group(1).splitlines() if l.strip()]
        struct["next_steps"] = bullets
    # Urgency
    m = re.search(r"Urgency:\s*(URGENT|ROUTINE)", text, re.I)
    if m:
        struct["urgency"] = m.group(1).upper()

    human_text = text  # full LLM output is already user-friendly
    return {"status":"ok", "source": source, "extracted_text": extracted, "human_text": human_text, "structured": struct, "raw_llm": text}
