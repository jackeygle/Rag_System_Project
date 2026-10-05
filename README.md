# Document Q&A

A single-user document library with grounded question answering. Upload PDF, TXT or Markdown files, search the whole library or selected documents, and inspect the original passages behind each answer.

## Start the web app

Python 3.12 is the tested environment.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements_streamlit.txt
cp .env.example .env
# Configure Azure OpenAI settings in .env (see below).
streamlit run streamlit_app.py
```

Files and vectors persist on the app instance. This is a personal app or a trusted shared library, not an authenticated multi-user service; sessions share the same files and index. Persistence requires a durable disk on hosted services.

## Azure OpenAI configuration

Set `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_ENDPOINT` (the HTTPS resource root), `AZURE_OPENAI_CHAT_DEPLOYMENT` and `AZURE_OPENAI_EMBEDDING_DEPLOYMENT` in `.env`. Both deployments must exist in the same Azure resource. The chat deployment serves your GPT model; the embedding deployment must serve an embedding model, such as `text-embedding-3-small`. Set `EMBEDDING_MODEL` to the actual embedding model behind that deployment.

The versioned Azure API defaults to `AZURE_OPENAI_API_VERSION=2024-10-21`; change it if your deployed model requires a different supported version. Deployment names are user-defined in Azure and may differ from model names. Do not use a ChatGPT website subscription token or an OpenAI direct API key here.

No Groq or Gemini key is required. Existing files are retained; switching to Azure creates a fresh embedding collection on the next synchronization and uses embedding quota for that initial pass. Changing the underlying model within the same deployment name requires also updating `EMBEDDING_MODEL` so the old vectors are not reused.

## Use

1. Upload PDF, TXT or MD files (20 MB limit per uploaded file).
2. Click **Save and index**. An existing file with the same name is replaced. Empty/unreadable uploads are rejected before replacement.
3. Choose **Entire library** or **Selected documents**.
4. Ask a self-contained question. Answers use numbered citations such as `[1]`.
5. Click a numbered source button beneath an answer (for example, **[1] notes.pdf**) to open its passage in the **Source reader**. The reader is closed until you choose a citation, and can be closed again. Earlier answers retain their own citations.
6. Use **Export** to download a Markdown copy of the answers and supporting passages.
7. To delete, expand its entry in the sidebar, check the confirmation box, then delete it. Its indexed passages are removed too.

The repository includes sample Markdown documents. Delete them through the UI if you want only your own files.

## Workspace interface

- Warm paper palette, editorial serif headings and geometric artwork
- Searchable file library with type, size and indexed status; file search sits above uploads
- Actual parsing/indexing stages and completed-file progress after uploads
- Top toolbar for new conversation, export and settings
- Clear whole-library / selected-document scope
- Prominent inline question composer beneath the search scope
- On-demand source reader with passage selection and a close button
- Distinct evidence-insufficient and operation-failed messages

On narrow screens, Streamlit provides a collapsible sidebar; the source reader stacks beneath the conversation rather than becoming a custom mobile drawer. Numbered source controls are native buttons below the answer, not clickable links inside the generated Markdown text.

## Reliability

- Only retrieved passages are passed as evidence; the prompt prohibits fallback to general knowledge.
- Empty retrieval returns an insufficient-evidence message without an LLM call.
- Missing or invalid citation IDs also return an insufficient-evidence message.
- Citation IDs identify the exact passages supplied to the model, not independent proof that every claim is correct. Read the passages to verify important answers.
- PDF page numbers are one-based physical page positions; they may differ from printed page labels.
- `MIN_RELEVANCE` filters cosine similarity results. The default `0.25` is a starting point, **not calibrated confidence**. Tune it on answerable and unanswerable questions from your documents.

## Persistent incremental indexing

Unchanged files are skipped using full-file SHA-256 hashes. Changed/new files are parsed and only new chunks are embedded; old chunks are retired after replacements have been written. A failed embedding call leaves old vectors available. A failed file may need to be repaired or removed before synchronization completes.

Changing the embedding model, chunk settings or index version creates a separate collection. This requires one initial re-index for the new configuration; old collections remain on disk rather than being silently deleted. Azure OpenAI embeddings use a separate collection from the old Gemini index; changing the resource or embedding deployment also starts a new collection.

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

Tests use local deterministic embeddings with a real persistent Chroma database and simulated model outputs. They cover indexing, deletion, failed replacement, retrieval scopes, source mapping, invalid citations and Streamlit startup. Real Azure OpenAI calls need your configured keys and are not part of these tests.

## Current limits

Scanned PDFs need OCR before import. DOCX/XLSX and OCR are not included yet. Conversation history is displayed but not used for query rewriting: ask self-contained follow-up questions. Hybrid retrieval, reranking, evidence entailment checks and an answer-quality benchmark are planned next steps.

## Hugging Face deployment

The target Space is https://huggingface.co/spaces/jackeygleee/jackeygleeeeee. The new interface uses Docker + Streamlit; it does not use the old Gradio entrypoint.

```bash
pip install huggingface_hub
hf auth login
python scripts/deploy_hf.py --dry-run
python scripts/deploy_hf.py
```

The deployment script uploads an explicit source allowlist and copies `README_HF.md` as the Space's `README.md`. It never uploads `.env`, local indexes or untracked user documents, and does not delete existing Space files or secrets. Set provider keys in Space Settings. Deployment uploads trigger a separate build; confirm that build before treating the new interface as live.
