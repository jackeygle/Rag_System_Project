"""Single-user document library and grounded Q&A interface."""
from pathlib import Path
import os
import tempfile
import streamlit as st
from config import DATA_DIR, EMBEDDING_MODEL, LLM_MODEL, validate_api_keys
from src.vector_store import sync_directory, list_indexed_documents
from src.document_loader import load_single_document
from src.retriever import create_retriever
from src.generator import create_rag_chain

st.set_page_config(page_title="Document Q&A", page_icon="📚", layout="wide")


def save_upload(upload):
    name = Path(upload.name.replace("\\", "/")).name
    if name.startswith(".") or Path(name).suffix.lower() not in {".pdf", ".txt", ".md"}:
        raise ValueError("Upload a PDF, TXT or Markdown file.")
    contents = upload.getvalue()
    if not contents:
        raise ValueError("The file is empty.")
    if len(contents) > 20 * 1024 * 1024:
        raise ValueError("Files must be at most 20 MB.")
    target = DATA_DIR / name
    with tempfile.NamedTemporaryFile(dir=DATA_DIR, suffix=target.suffix, delete=False) as handle:
        handle.write(contents)
        temp = Path(handle.name)
    try:
        docs = load_single_document(temp)
        if not any(doc.page_content.strip() for doc in docs):
            raise ValueError("No readable text was found. Scanned PDFs need OCR before upload.")
        os.replace(temp, target)
    finally:
        temp.unlink(missing_ok=True)
    return name


def main():
    st.title("📚 Document Q&A")
    st.caption("Ask questions about your documents and inspect the passages behind each answer.")
    if "messages" not in st.session_state:
        st.session_state.messages = []
    valid, missing = validate_api_keys()
    if not valid:
        st.warning("Configure these API keys in .env before indexing or asking questions: " + ", ".join(missing))
    with st.sidebar:
        st.header("Document library")
        st.caption("PDF, TXT and Markdown · up to 20 MB per file. Files with the same name are replaced.")
        uploads = st.file_uploader("Add documents", type=["pdf", "txt", "md"], accept_multiple_files=True)
        if st.button("Save and index", disabled=not uploads or not valid, use_container_width=True):
            try:
                names = [save_upload(upload) for upload in uploads]
                with st.spinner("Indexing new and changed documents…"):
                    sync_directory(DATA_DIR)
                st.success("Indexed: " + ", ".join(names))
            except Exception as error:
                st.error(str(error))
        files = sorted(path for path in DATA_DIR.rglob("*") if path.is_file() and path.suffix.lower() in {".pdf", ".txt", ".md"})
        labels = {str(path.resolve()): str(path.relative_to(DATA_DIR)) for path in files}
        if files:
            selected_delete = st.selectbox("Remove a file", list(labels), format_func=labels.get)
            confirmed = st.checkbox("Delete this file and its indexed passages")
            if st.button("Delete selected file", disabled=not confirmed or not valid):
                try:
                    Path(selected_delete).unlink()
                    sync_directory(DATA_DIR)
                    st.rerun()
                except Exception as error:
                    st.error(str(error))
        else:
            st.info("Your library is empty. Add a document to get started.")
        st.divider()
        if st.button("Clear conversation", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
        st.caption("Stored on this app instance. This version is for one user or a trusted shared library.")

    store = None
    if valid:
        try:
            # File hashes avoid repeated parsing and embedding on Streamlit reruns.
            store = sync_directory(DATA_DIR)
        except Exception as error:
            st.error(f"Index synchronization failed: {error}")
    indexed = list_indexed_documents() if valid else []
    options = {record["source"]: record["file_name"] for record in indexed}
    scope_mode = st.radio("Search scope", ["Entire library", "Selected documents"], horizontal=True)
    chosen = None
    if scope_mode == "Selected documents":
        chosen = st.multiselect("Documents to search", list(options), format_func=options.get)
        if not chosen:
            st.info("Select at least one document to ask a question.")
    st.caption(f"{len(indexed)} indexed documents · {LLM_MODEL} · {EMBEDDING_MODEL}")
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            for source in message.get("sources", []):
                label = f"[{source['id']}] {source['file_name']}"
                if source["page"] is not None:
                    label += f" · page {source['page']}"
                with st.expander(label):
                    st.caption(source["source"])
                    st.text(source["excerpt"])
    question = st.chat_input("Ask a question about your documents", disabled=store is None or not indexed or chosen == [])
    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        try:
            retriever = create_retriever(store, sources=chosen)
            chain = create_rag_chain(retriever)
            with st.spinner("Searching documents and preparing an answer…"):
                result = chain.invoke(question)
            st.session_state.messages.append({"role": "assistant", "content": result["answer"], "sources": result["sources"]})
        except Exception as error:
            st.session_state.messages.append({"role": "assistant", "content": f"Could not complete this question: {error}"})
        st.rerun()
    st.caption("Answers use retrieved passages. Citation numbering is validated; factual support still needs your review. Follow-up questions should be self-contained in this version.")


if __name__ == "__main__":
    main()
