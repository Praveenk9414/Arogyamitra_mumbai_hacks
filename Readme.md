# Arogyamitra — AI Medical Assistant (MVP)

This repository contains a hackathon-ready MVP for **Arogyamitra** — an agentic AI assistant for patient reports, hospital bills, and price comparison. It includes:

- Gradio UI (`app.py`) for Procedure Validator, Price Compare, Bill Auditor and Report Q&A.
- Streamlit RAG demo (`app_multisession_rag.py`) for multi-session retrieval-augmented generation.
- `tools/` modules: `validate_procedure.py`, `audit_bill.py`, `price_compare.py`, `rag_interface.py`.
- `extractors/` module: `text_extractor.py` (pdfplumber-based extractor & chunker).
- `agent.py` — aggregator agent (ingest files, summarize, answer user questions).
- Local LLM wrapper `llm.py` that calls Ollama CLI (or falls back to other mechanisms).
- `data/` sample JSONs: `guidelines.json`, `hospitals.json`, `billing_rules.json`.

---

## Quick repo layout
Arogyamitra/
├── app.py # Gradio UI (MVP)
├── app_multisession_rag.py # Streamlit RAG multi-session demo
├── agent.py # Aggregation agent (ingest + ask)
├── llm.py # Ollama CLI wrapper (local LLM)
├── requirements.txt
├── README.md
├── data/
│ ├── guidelines.json
│ ├── hospitals.json
│ └── billing_rules.json
├── tools/
│ ├── validate_procedure.py
│ ├── audit_bill.py
│ ├── price_compare.py
│ └── rag_interface.py
└── extractors/
└── text_extractor.py





Set Environment Variables:

# Ollama model (default used by llm.py)
export OLLAMA_MODEL="koesn/llama3-8b-instruct:latest"
export OLLAMA_TIMEOUT_SECS=60

# optional: if using OpenAI fallback
export OPENAI_API_KEY="sk-..."

# Optional: chroma / embedding settings (if you want to override defaults)
export DEFAULT_EMBEDDING_MODEL="nomic-embed-text:latest"
export DEFAULT_LLAMA_MODEL="koesn/llama3-8b-instruct:latest"






Deploy:

# from repo root
python app.py
# Open the URL printed (usually http://127.0.0.1:7860)

