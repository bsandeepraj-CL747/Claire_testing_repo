# AI Document Chat (RAG)

A small, self-contained Retrieval-Augmented Generation (RAG) web app.
Upload a document (PDF, DOCX, TXT, or Markdown) and chat with it.

## Run

```bash
python3 server.py
```

Then open http://127.0.0.1:8000 in your browser.

Optional: set `OPENAI_API_KEY` to get LLM-generated answers. Without it, the app
returns extractive answers built from the most relevant passages of your document.

```bash
OPENAI_API_KEY=sk-... python3 server.py
```

## Requirements

- Python 3.9+
- `pypdf` and `python-docx` (optional, only needed for PDF/DOCX). Install with:
  `pip install pypdf python-docx`

## Notes

- Documents are processed in memory; nothing is persisted to disk.
- Max upload size: 10 MB.
