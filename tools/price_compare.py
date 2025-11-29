# tools/price_compare.py
"""
Price Compare tool using hospital data from data/hospitals.json.

Features:
- load hospitals.json
- normalize procedure names
- get quotes for a given procedure and city
- given a bill (text or file) attempt to detect procedure keywords and compare billed amounts to expected hospital package
"""

import os
import json
import re
from typing import Dict, Any, List, Optional

DATA_HOSPITALS = "data/hospitals.json"

# small fallback if data file missing
DEFAULT_HOSPITALS = []

# --- data loading ---
def load_hospitals(path: str = DATA_HOSPITALS) -> List[Dict[str, Any]]:
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    return data
        except Exception:
            pass
    return DEFAULT_HOSPITALS

# --- normalize ---
def normalize_proc_name(name: str) -> str:
    if not name:
        return ""
    s = name.strip().lower()
    s = s.replace("-", " ").replace("_", " ")
    # common synonyms
    syn = {
        "appendicitis": "appendectomy",
        "appendix removal": "appendectomy",
        "appendix": "appendectomy",
        "cataract": "cataract_surgery",
        "phaco": "cataract_surgery",
        "meniscus": "meniscus"
    }
    for k, v in syn.items():
        if k in s:
            return v
    return re.sub(r"\s+", "_", s)

# --- find quotes ---
def find_quotes(proc_key: str, city: Optional[str] = None) -> List[Dict[str, Any]]:
    hospitals = load_hospitals()
    results = []
    for h in hospitals:
        if city and h.get("city") and h["city"].lower() != city.lower():
            continue
        proc = h.get("procedures", {}).get(proc_key)
        if proc:
            results.append({
                "hospital_id": h.get("id"),
                "hospital": h.get("name"),
                "city": h.get("city"),
                "procedure_key": proc_key,
                "expected_price": proc.get("expected_price"),
                "breakdown": proc.get("breakdown", {})
            })
    results = sorted(results, key=lambda x: x.get("expected_price", 10**9))
    return results

def get_quotes(procedure: str, city: Optional[str] = None) -> Dict[str, Any]:
    proc_key = normalize_proc_name(procedure)
    quotes = find_quotes(proc_key, city)
    if not quotes:
        return {
            "status": "not_found",
            "procedure": proc_key,
            "city": city,
            "quotes": [],
            "human_text": f"No quotes found for '{procedure}' in {city or 'any city'}."
        }
    human_lines = []
    for q in quotes:
        br = ", ".join([f"{k}: ₹{v}" for k, v in (q["breakdown"] or {}).items()])
        human_lines.append(f"{q['hospital']} ({q['city']}) — ₹{q['expected_price']}. Breakdown: {br}")
    return {"status": "ok", "procedure": proc_key, "city": city, "quotes": quotes, "human_text": "\n".join(human_lines)}

# --- bill parsing helpers ---
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

KNOWN_PROC_KEYWORDS = {
    "appendectomy": ["appendectomy", "appendix", "appendicitis"],
    "cataract_surgery": ["cataract", "phaco", "i o l", "iol", "lens"],
    "meniscus": ["meniscus", "meniscal", "arthroscopy"],
    "cholecystectomy": ["chole", "cholecystectomy", "gallbladder"],
    "hernia": ["hernia", "inguinal hernia"]
}

AMOUNT_RE = re.compile(r"(?:Rs\.?|₹)\s?([0-9,]+)")

def parse_amount(s: str) -> Optional[int]:
    if not s:
        return None
    s2 = s.replace(",", "").strip()
    try:
        return int(s2)
    except Exception:
        return None

def parse_bill_for_procedures(bill_text: str) -> List[str]:
    text = bill_text.lower()
    found = []
    for key, keywords in KNOWN_PROC_KEYWORDS.items():
        for kw in keywords:
            if kw in text:
                found.append(key)
                break
    return list(dict.fromkeys(found))

def extract_line_amounts(bill_text: str) -> List[Dict[str, Any]]:
    lines = [l.strip() for l in bill_text.splitlines() if l.strip()]
    results = []
    for l in lines:
        m = AMOUNT_RE.search(l)
        if m:
            amt = parse_amount(m.group(1))
            if amt is not None:
                results.append({"line": l, "amount": amt})
    return results

# --- public: get quotes by examining a bill file/text ---
def get_quotes_from_bill(bill_text_or_path: str, city: Optional[str] = None) -> Dict[str, Any]:
    if os.path.exists(bill_text_or_path):
        text = _read_text_from_file(bill_text_or_path)
    else:
        text = bill_text_or_path

    detected = parse_bill_for_procedures(text)
    if not detected:
        return {"status": "not_found", "human_text": "No known procedure keywords detected in bill text.", "procedures": [], "quotes": {}}

    quotes_for = {}
    comparison = []
    lines_with_amounts = extract_line_amounts(text)
    for proc in detected:
        quotes = find_quotes(proc, city)
        quotes_for[proc] = quotes
        # compare billed sums with expected price (best quote)
        if quotes:
            best_expected = quotes[0]["expected_price"]
            # heuristics: find lines near procedure keyword or any large amounts
            matched_amounts = []
            # find amounts in text near keyword (simple approach: search for lines containing keyword)
            for l in lines_with_amounts:
                if proc.replace("_", " ") in l["line"].lower() or any(k in l["line"].lower() for k in KNOWN_PROC_KEYWORDS.get(proc, [])):
                    matched_amounts.append(l["amount"])
            # fallback: take largest billed line if none near keyword
            if not matched_amounts and lines_with_amounts:
                matched_amounts = [max(l["amount"] for l in lines_with_amounts)]
            billed = matched_amounts[0] if matched_amounts else None
            flagged = False
            pct_over = None
            if billed:
                pct_over = round((billed - best_expected) / best_expected * 100, 2) if best_expected else None
                # load billing rules threshold if present
                brules = {}
                try:
                    with open("data/billing_rules.json", "r", encoding="utf-8") as f:
                        brules = json.load(f)
                except Exception:
                    brules = {}
                threshold = brules.get("overcharge_pct_threshold", 25)
                if pct_over is not None and pct_over > threshold:
                    flagged = True
            comparison.append({
                "procedure": proc,
                "expected_price": best_expected,
                "billed_amount_detected": billed,
                "pct_over_expected": pct_over,
                "flagged_overcharge": flagged,
                "quotes": quotes
            })
    human_text = f"Detected procedures: {', '.join(detected)}"
    return {"status": "ok", "procedures": detected, "quotes": quotes_for, "comparison": comparison, "human_text": human_text}
