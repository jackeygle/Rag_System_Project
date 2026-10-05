# Document Q&A

A single-user document library with grounded question answering. Upload PDF, TXT or Markdown files, search the whole library or selected documents, and inspect the original passages behind each answer.

## Start the web app

Python 3.12 is the tested environment.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements_streamlit.txt
cp .env.example .env
# Add GOOGLE_API_KEY and GROQ_API_KEY to .env.
streamlit run streamlit_app.py
```

Files and vectors persist on the app instance. This is a personal app or a trusted shared library, not an authenticated multi-user service; sessions share the same files and index. Persistence requires a durable disk on hosted services.

## Use

1. Upload PDF, TXT or MD files (20 MB limit per uploaded file).
2. Click **Save and index**. An existing file with the same name is replaced. Empty/unreadable uploads are rejected before replacement.
3. Choose **Entire library** or **Selected documents**.
4. Ask a self-contained question. Answers use numbered citations such as `[1]`.
5. Read the cited passages in the **Source reader** beside the conversation. Choose **Inspect sources** on an earlier answer to switch the reader.
6. Use **Export conversation** to download a Markdown copy of the answers and supporting passages.
7. To delete, select a file in the sidebar, check the confirmation box, then delete it. Its indexed passages are removed too.

The repository includes sample Markdown documents. Delete them through the UI if you want only your own files.

## Reliability

- Only retrieved passages are passed as evidence; the prompt prohibits fallback to general knowledge.
- Empty retrieval returns an insufficient-evidence message without an LLM call.
- Missing or invalid citation IDs also return an insufficient-evidence message.
- Citation IDs identify the exact passages supplied to the model, not independent proof that every claim is correct. Read the passages to verify important answers.
- PDF page numbers are one-based physical page positions; they may differ from printed page labels.
- `MIN_RELEVANCE` filters cosine similarity results. The default `0.25` is a starting point, **not calibrated confidence**. Tune it on answerable and unanswerable questions from your documents.

## Persistent incremental indexing

Unchanged files are skipped using full-file SHA-256 hashes. Changed/new files are parsed and only new chunks are embedded; old chunks are retired after replacements have been written. A failed embedding call leaves old vectors available. A failed file may need to be repaired or removed before synchronization completes.

Changing the embedding model, chunk settings or index version creates a separate collection. This requires one initial re-index for the new configuration; old collections remain on disk rather than being silently deleted. The former `text-embedding-004` default has been replaced with configurable `models/gemini-embedding-001`.

For one application process, index mutations are serialized with a lock. Multiple independent workers sharing the same disk are not supported by this implementation.

## CLI

```bash
pip install -r requirements.txt
python main.py --index
python main.py --index --url https://example.com
python main.py --query "What do the documents say about retrieval?"
python main.py
```

The CLI and legacy Gradio app reuse the same retrieval/generation/index modules. Library upload and source expanders are provided by the Streamlit interface. CLI/Gradio display citation IDs and source names in text.

## Tests (no provider keys or paid API calls)

```bash
pip install pytest
python -m pytest -q tests
```

Tests use local deterministic embeddings with a real persistent Chroma database and simulated model outputs. They cover indexing, deletion, failed replacement, retrieval scopes, source mapping, invalid citations and Streamlit startup. Real Gemini/Groq calls need your configured keys and are not part of these tests.

## Current limits

Scanned PDFs need OCR before import. DOCX/XLSX and OCR are not included yet. Conversation history is displayed but not used for query rewriting: ask self-contained follow-up questions. Hybrid retrieval, reranking, evidence entailment checks and an answer-quality benchmark are planned next steps.
