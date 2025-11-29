# tools/audit_bill.py
"""
Audit a hospital bill using:
- rule-based checks (duplicates, large charges)
- cross-check against hospital expected prices (data/hospitals.json)
- billing rules from data/billing_rules.json
- optional local LLM to draft dispute email (calls llm.safe_llm_summary)
"""

import os
import re
import json
from typing import Dict, Any, List, Optional
from llm import safe_llm_summary
from tools.price_compare import parse_bill_for_procedures, extract_line_amounts, get_quotes_from_bill

AMOUNT_RE = re.compile(r"(?:Rs\.?|₹)\s?([0-9,]+)")

def _read_text_from_file(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        try:
            import fitz
            doc = fitz.open(path)
            pages = [p.get_text("text") for p in doc]
            doc.close()
            return "\n\n".join(pages)
        except Exception:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                return f.read()
    else:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()

def find_duplicate_lines(text: str, tolerance_chars: int = 6) -> List[str]:
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    seen = {}
    dups = []
    for l in lines:
        key = l[:tolerance_chars].lower()
        seen[key] = seen.get(key, 0) + 1
        if seen[key] == 2:
            dups.append(l)
    return dups

def find_large_charges(text: str, threshold: int = 20000) -> List[Dict[str, Any]]:
    results = []
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    for l in lines:
        m = AMOUNT_RE.search(l.replace(",", ""))
        if m:
            try:
                amt = int(m.group(1))
                if amt >= threshold:
                    results.append({"line": l, "amount": amt})
            except:
                continue
    return results

def load_billing_rules(path: str = "data/billing_rules.json") -> Dict[str, Any]:
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    # defaults
    return {"overcharge_pct_threshold": 25, "large_single_charge_threshold": 20000, "duplicate_line_match_tolerance_chars": 6}

def audit_bill(bill_text_or_path: str, use_llm: bool = True) -> Dict[str, Any]:
    # extract text
    if os.path.exists(bill_text_or_path):
        try:
            text = _read_text_from_file(bill_text_or_path)
        except Exception as e:
            return {"status":"error", "error": str(e)}
    else:
        text = bill_text_or_path

    brules = load_billing_rules()
    duptol = brules.get("duplicate_line_match_tolerance_chars", 6)
    threshold = brules.get("large_single_charge_threshold", 20000)
    over_pct = brules.get("overcharge_pct_threshold", 25)

    flags = []

    # 1) duplicate lines
    dups = find_duplicate_lines(text, tolerance_chars=duptol)
    for d in dups:
        flags.append({"type":"duplicate", "line": d, "reason": "Potential duplicate charge line"})

    # 2) large single charges
    large = find_large_charges(text, threshold=threshold)
    for l in large:
        flags.append({"type":"large_charge", "line": l["line"], "amount": l["amount"], "reason": "Single large charge"})

    # 3) cross-check bill against hospital package prices (if procedures found)
    procedures = parse_bill_for_procedures(text)
    if procedures:
        comparison = get_quotes_from_bill(text)
        # each item in comparison['comparison'] has pct_over_expected and flagged_overcharge
        for comp in comparison.get("comparison", []):
            if comp.get("flagged_overcharge"):
                flags.append({
                    "type":"overcharge",
                    "procedure": comp.get("procedure"),
                    "expected_price": comp.get("expected_price"),
                    "billed": comp.get("billed_amount_detected"),
                    "pct_over": comp.get("pct_over_expected"),
                    "reason":"Billed amount exceeds expected package price by threshold"
                })

    # build a short structured summary
    summary = f"Found {len(flags)} potential issues."

    # LLM: craft dispute email referencing flagged items
    dispute_email = None
    raw_llm = None
    if use_llm:
        # prepare a concise context for the model
        brief_flags = []
        for f in flags:
            if f.get("type") == "duplicate":
                brief_flags.append(f"Duplicate line: '{f.get('line')}'")
            elif f.get("type") == "large_charge":
                brief_flags.append(f"Large charge: '{f.get('line')}' amount ₹{f.get('amount')}")
            elif f.get("type") == "overcharge":
                brief_flags.append(f"Overcharge: procedure {f.get('procedure')} billed ₹{f.get('billed')}, expected ₹{f.get('expected_price')} (~{f.get('pct_over')}% over)")

        prompt = (
            "You are a hospital bill auditor assistant. Draft a short, polite dispute email to the hospital billing team.\n"
            "Include: 1) reference to flagged items (list below). 2) Request for corrected, itemized invoice and justification for charges. 3) Ask for contact person and expected turnaround.\n\n"
            f"Flagged items:\n- " + "\n- ".join(brief_flags) + "\n\n"
            "Return only the email text (no additional commentary)."
        )
        try:
            raw_llm = safe_llm_summary(prompt)
            dispute_email = raw_llm
        except Exception as e:
            dispute_email = f"LLM error generating email: {e}"
            raw_llm = str(e)

    return {
        "status":"ok",
        "flags": flags,
        "summary": summary,
        "dispute_email": dispute_email,
        "raw_llm": raw_llm,
        "extracted_text": text
    }
