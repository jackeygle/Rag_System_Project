---
title: Document Q&A Workspace
emoji: 📚
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
license: mit
---

# Document Q&A Workspace

Upload PDF, TXT or Markdown documents, ask questions across your library or selected files, and open numbered citations to inspect the original passages.

This Space runs the Streamlit workspace inside Docker, on port 7860.

## Space configuration

Set these in **Settings → Variables and secrets**:

- Secret `GOOGLE_API_KEY`: Gemini embeddings key
- Secret `GROQ_API_KEY`: Groq answer model key
- Optional variable `EMBEDDING_MODEL`: defaults to `models/gemini-embedding-001`
- Optional variable `LLM_MODEL`: defaults to `llama-3.3-70b-versatile`
- Optional variable `MIN_RELEVANCE`: defaults to `0.25`; tune on your document collection
- Optional variable `RAG_STORAGE_DIR`: a writable durable directory, if storage is attached

Without keys, the UI starts and explains what configuration is missing. Keep keys in Space secrets, never in repository files.

This is a personal or trusted shared workspace. Visitors share its uploaded files and index. Default container storage is ephemeral across Space restarts/rebuilds; attach durable storage and set `RAG_STORAGE_DIR` if files must persist.

Scanned PDFs need OCR before import. References identify retrieved passages, not automatic proof that every claim is supported.

Source: https://github.com/jackeygle/Rag_System_Project
