# tools/cost_negotiator.py
from typing import Dict, Any
from llm import safe_llm_summary

def negotiation_script(procedure: str, quotes: list) -> Dict[str, Any]:
    """
    Build a short negotiation script from quotes.
    If local LLM is available, ask it to craft a 3-line script + 2 key questions.
    """
    prompt = (
        "You are a negotiation assistant. Given these hospital quotes, create 3 concise lines a patient can say on call to request an all-inclusive binding quote, "
        "and 2 questions to confirm no hidden charges.\n\n"
        f"Procedure: {procedure}\nQuotes: {quotes}\n\nReturn:\nScript:\n- line1\n- line2\n- line3\n\nQuestions:\n- q1\n- q2"
    )
    try:
        out = safe_llm_summary(prompt)
        return {"status": "ok", "script": out}
    except Exception as e:
        # fallback simple template
        script = (
            "Hello, I need an all-inclusive, fixed-price quote for {procedure}. "
            "Can you confirm the price includes surgeon, OT, implants, stay, and medicines?\n"
            "If you can match the lowest competing quote (₹...), I will book with you.\n\n"
            "Questions:\n- Is this an all-inclusive price?\n- Are there any additional consumable or surgeon fees not listed?"
        ).format(procedure=procedure)
        return {"status": "ok", "script": script}
