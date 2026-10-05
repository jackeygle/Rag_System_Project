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


def conversation_export(messages):
    lines = ["# Document Q&A conversation", ""]
    for message in messages:
        lines.extend(["## " + message["role"].title(), message["content"], ""])
        for source in message.get("sources", []):
            page = f" · page {source['page']}" if source["page"] is not None else ""
            lines.extend([f"[{source['id']}] {source['file_name']}{page}", source["excerpt"], ""])
    return "\n".join(lines)


def main():
    stylesheet = Path(__file__).parent / "assets" / "workspace.css"
    st.markdown("<style>" + stylesheet.read_text() + "</style>", unsafe_allow_html=True)
    if "messages" not in st.session_state:
        st.session_state.messages = []
    valid, missing = validate_api_keys()
    notification = st.session_state.pop("notification", None)
    if notification:
        st.toast(notification)

    with st.sidebar:
        st.markdown('<div class="workspace-brand">▣ <span>Document</span> workspace</div>', unsafe_allow_html=True)
        st.caption("Your files. Answers you can check.")
        st.divider()
        st.subheader("Add documents")
        uploads = st.file_uploader("PDF, TXT or Markdown", type=["pdf", "txt", "md"], accept_multiple_files=True,
                                   help="Up to 20 MB per file. Same-name files replace the existing version.")
        if st.button("Add to library", type="primary", disabled=not uploads or not valid, use_container_width=True):
            try:
                names = [save_upload(upload) for upload in uploads]
                with st.spinner("Preparing your documents…"):
                    sync_directory(DATA_DIR)
                st.session_state.notification = "Added: " + ", ".join(names)
                st.rerun()
            except Exception as error:
                st.error(str(error))
        files = sorted(path for path in DATA_DIR.rglob("*") if path.is_file() and path.suffix.lower() in {".pdf", ".txt", ".md"})
        labels = {str(path.resolve()): str(path.relative_to(DATA_DIR)) for path in files}
        st.subheader("Your library")
        st.caption(f"{len(files)} file{'s' if len(files) != 1 else ''}")
        for path in files:
            st.write("▤ " + str(path.relative_to(DATA_DIR)))
        if files:
            with st.expander("Manage files"):
                selected_delete = st.selectbox("File to remove", list(labels), format_func=labels.get)
                confirmed = st.checkbox("Delete the file and its indexed passages")
                if st.button("Delete file", disabled=not confirmed or not valid, use_container_width=True):
                    try:
                        Path(selected_delete).unlink()
                        sync_directory(DATA_DIR)
                        st.session_state.notification = "File removed"
                        st.rerun()
                    except Exception as error:
                        st.error(str(error))
        else:
            st.info("Add your first document to get started.")
        st.divider()
        if st.button("New conversation", use_container_width=True):
            st.session_state.messages = []
            st.session_state.pop("source_answer", None)
            st.rerun()
        if st.session_state.messages:
            st.download_button("Export conversation", conversation_export(st.session_state.messages),
                               "document-conversation.md", "text/markdown", use_container_width=True)
        with st.expander("About this workspace"):
            st.caption("Files persist on this app instance. This is a personal or trusted shared library; it does not separate files by user.")
            st.caption(f"Answer model: {LLM_MODEL}\n\nEmbedding model: {EMBEDDING_MODEL}")
            st.caption("Scanned PDFs need OCR before import. Follow-up questions should be self-contained.")

    st.markdown('<div class="workspace-kicker">DOCUMENT INTELLIGENCE</div>', unsafe_allow_html=True)
    st.title("Find answers in your documents.")
    st.caption("Ask a question, then follow the citations back to the original text.")
    if not valid:
        st.warning("Add " + ", ".join(missing) + " to your .env file to connect the workspace.")
    store = None
    if valid:
        try:
            with st.spinner("Checking your library…"):
                store = sync_directory(DATA_DIR)
        except Exception as error:
            st.error(f"Could not prepare the library: {error}")
    indexed = list_indexed_documents() if valid else []
    options = {record["source"]: record["file_name"] for record in indexed}
    with st.container(border=True):
        st.caption("ANSWER FROM")
        scope_mode = st.radio("Search scope", ["Entire library", "Selected documents"], horizontal=True, label_visibility="collapsed")
        chosen = None
        if scope_mode == "Selected documents":
            chosen = st.multiselect("Documents to search", list(options), format_func=options.get, placeholder="Choose documents…")
            if not chosen:
                st.info("Choose at least one document to enable questions.")
        count = len(indexed) if chosen is None else len(chosen)
        st.caption(f"{count} document{'s' if count != 1 else ''} in scope · Answers include source references")

    conversation, evidence = st.columns([2.1, 1], gap="large")
    pending = None
    with conversation:
        st.subheader("Conversation")
        if not st.session_state.messages:
            st.markdown('<div class="workspace-welcome"><h2>What would you like to know?</h2><p>Find a detail, compare what your documents say, or ask for an explanation with supporting passages.</p></div>', unsafe_allow_html=True)
            st.caption("Example questions — edit them to fit your files")
            for prompt in ["What do these documents say about retrieval?", "How is reinforcement learning different from supervised learning?", "What limitations are discussed in the documents?"]:
                if st.button(prompt, disabled=store is None or not indexed or chosen == [], use_container_width=True):
                    pending = prompt
        for index, message in enumerate(st.session_state.messages):
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                if message.get("sources"):
                    st.caption("Sources: " + " · ".join(f"[{source['id']}] {source['file_name']}" for source in message["sources"]))
                    if st.button("Inspect sources", key=f"inspect_{index}"):
                        st.session_state.source_answer = index
                        st.rerun()
        question = st.chat_input("Ask about the documents in scope…", disabled=store is None or not indexed or chosen == [])
        if pending:
            question = pending
        if question:
            st.session_state.messages.append({"role": "user", "content": question})
            try:
                chain = create_rag_chain(create_retriever(store, sources=chosen))
                with st.spinner("Finding supporting passages…"):
                    result = chain.invoke(question)
                st.session_state.messages.append({"role": "assistant", "content": result["answer"], "sources": result["sources"]})
                st.session_state.source_answer = len(st.session_state.messages) - 1
            except Exception as error:
                st.session_state.messages.append({"role": "assistant", "content": f"Could not complete this question: {error}"})
                st.session_state.source_answer = len(st.session_state.messages) - 1
            st.rerun()

    with evidence:
        st.subheader("Source reader")
        source_index = st.session_state.get("source_answer", -1)
        selected = st.session_state.messages[source_index] if 0 <= source_index < len(st.session_state.messages) else {}
        sources = selected.get("sources", [])
        if sources:
            st.caption("Evidence for the selected answer")
            with st.container(border=True):
                st.caption("QUESTION")
                if source_index > 0:
                    st.write(st.session_state.messages[source_index - 1]["content"])
            for source in sources:
                label = f"[{source['id']}] {source['file_name']}"
                if source["page"] is not None:
                    label += f" · page {source['page']}"
                with st.expander(label, expanded=len(sources) == 1):
                    st.text(source["excerpt"])
                    st.caption("Source: " + source["source"])
        else:
            with st.container(border=True):
                st.write("▤ Original passages appear here")
                st.caption("Ask a question to see its references. For an earlier answer, choose Inspect sources.")
                if selected:
                    st.caption("This answer has no cited evidence.")
        st.caption("References identify supplied passages. Review the text to verify important claims.")


if __name__ == "__main__":
    main()
