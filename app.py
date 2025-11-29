# app.py
import gradio as gr
from tools.translate import translate_report_input
from tools.validate_procedure import validate_procedure_tool
from tools.audit_bill import audit_bill
from tools.price_compare import get_quotes_from_bill, get_quotes

def translate_from_file(file):
    """
    file: gr.File (temp path)
    """
    if file is None:
        return "Upload a PDF or paste text."
    path = file.name
    res = translate_report_input(path)
    if res.get("status") == "ok":
        return res.get("human_text", "No output")
    else:
        return f"Error: {res.get('error')}"

def translate_from_text(text):
    if not text:
        return "Paste a report text or upload a file."
    res = translate_report_input(text)
    return res.get("human_text", "No output")

def validate_from_file(file):
    if file is None:
        return "Upload a report file or paste text."
    path = file.name
    res = validate_procedure_tool(path)
    if res.get("status") == "ok":
        out = f"Necessity: {res.get('necessity')}\nRationale: {res.get('rationale')}\n\nAlternatives:\n"
        out += "\n".join([f"- {a}" for a in res.get("alternatives",[])])
        return out
    return f"Error: {res.get('error','unknown')}"

def validate_from_text(text):
    if not text:
        return "Paste a report excerpt or procedure name."
    res = validate_procedure_tool(text)
    if res.get("status") == "ok":
        out = f"Necessity: {res.get('necessity')}\nRationale: {res.get('rationale')}\n\nAlternatives:\n"
        out += "\n".join([f"- {a}" for a in res.get("alternatives",[])])
        return out
    return f"Error: {res.get('error','unknown')}"

def audit_bill_from_file(file):
    if file is None:
        return "Upload bill file or paste text."
    path = file.name
    res = audit_bill(path, use_llm=True)
    return res.get("dispute_email", "No dispute email generated"), res.get("flags", [])

def audit_bill_from_text(text):
    if not text:
        return "Paste bill text."
    res = audit_bill(text, use_llm=True)
    return res.get("dispute_email", "No dispute generated"), res.get("flags", [])

def price_from_bill_file(file):
    if file is None:
        return "Upload bill file or paste bill text."
    path = file.name
    res = get_quotes_from_bill(path)
    return res.get("human_text", "No procedures detected"), res.get("quotes", {})

def price_from_bill_text(text):
    if not text:
        return "Paste bill text."
    res = get_quotes_from_bill(text)
    return res.get("human_text","No procedures detected"), res.get("quotes", {})

def price_manual(procedure, city):
    return get_quotes(procedure, city)["human_text"]

with gr.Blocks(title="Arogyamitra - Local LLM + File Upload Demo") as demo:
    gr.Markdown("# Arogyamitra — Upload-driven demo")

    with gr.Tab("Translate Report (User)"):
        gr.Markdown("Upload a report PDF (scanned selectable PDF) or paste report text.")
        with gr.Row():
            report_file = gr.File(label="Upload report (PDF/TXT)")
            translate_file_btn = gr.Button("Translate uploaded file")
        translate_file_out = gr.Textbox(lines=10)
        translate_file_btn.click(translate_from_file, [report_file], translate_file_out)

        gr.Markdown("Or paste report text:")
        report_text = gr.Textbox(lines=8, placeholder="Paste report text here...")
        translate_text_btn = gr.Button("Translate pasted text")
        translate_text_out = gr.Textbox(lines=10)
        translate_text_btn.click(translate_from_text, [report_text], translate_text_out)

    with gr.Tab("Procedure Validator (User)"):
        gr.Markdown("Upload report or paste snippet (Validator will use LLM).")
        val_file = gr.File(label="Upload report (PDF/TXT)")
        val_file_btn = gr.Button("Validate uploaded file")
        val_file_out = gr.Textbox(lines=8)
        val_file_btn.click(validate_from_file, [val_file], val_file_out)

        gr.Markdown("Or paste procedure / excerpt:")
        val_text = gr.Textbox(lines=4)
        val_text_btn = gr.Button("Validate pasted text")
        val_text_out = gr.Textbox(lines=8)
        val_text_btn.click(validate_from_text, [val_text], val_text_out)

    with gr.Tab("Hospital Bill (Hospital)"):
        gr.Markdown("Hospital: upload final bill (PDF/TXT) — it will be used by both Audit and Price Compare.")
        bill_file = gr.File(label="Upload bill (PDF/TXT)")
        bill_audit_btn = gr.Button("Audit uploaded bill (dispute email)")
        bill_audit_email = gr.Textbox(lines=8)
        bill_audit_flags = gr.JSON()
        bill_audit_btn.click(audit_bill_from_file, [bill_file], [bill_audit_email, bill_audit_flags])

        gr.Markdown("Price Compare from uploaded bill (detect procedures & show mock quotes):")
        bill_price_btn = gr.Button("Get quotes from uploaded bill")
        bill_price_out = gr.Textbox(lines=8)
        bill_price_json = gr.JSON()
        bill_price_btn.click(price_from_bill_file, [bill_file], [bill_price_out, bill_price_json])

        gr.Markdown("Or paste bill text to analyze:")
        bill_text = gr.Textbox(lines=8)
        bill_text_audit_btn = gr.Button("Audit pasted bill")
        bill_text_audit_out = gr.Textbox(lines=8)
        bill_text_flags = gr.JSON()
        bill_text_audit_btn.click(audit_bill_from_text, [bill_text], [bill_text_audit_out, bill_text_flags])

        bill_text_price_btn = gr.Button("Get quotes from pasted bill")
        bill_text_price_out = gr.Textbox(lines=8)
        bill_text_price_json = gr.JSON()
        bill_text_price_btn.click(price_from_bill_text, [bill_text], [bill_text_price_out, bill_text_price_json])

        gr.Markdown("Manual price query:")
        manual_proc = gr.Textbox(placeholder="appendectomy")
        manual_city = gr.Textbox(value="Bangalore")
        manual_price_btn = gr.Button("Get quotes")
        manual_price_out = gr.Textbox(lines=6)
        manual_price_btn.click(price_manual, [manual_proc, manual_city], manual_price_out)

    gr.Markdown("**Note:** This demo uses a local LLM (Ollama) for translations and structured outputs. Uploads use temporary paths provided by Gradio.")

if __name__ == "__main__":
    demo.launch(share=False)
