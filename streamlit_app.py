"""Single-user document library and grounded Q&A interface."""
from pathlib import Path
import os
import tempfile
import streamlit as st
from config import DATA_DIR, EMBEDDING_MODEL, LLM_MODEL, validate_api_keys
from src.vector_store import sync_directory, list_indexed_documents, file_digest
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


def reset_conversation():
    st.session_state.messages = []
    for key in ("source_answer", "source_id", "source_reader_open"):
        st.session_state.pop(key, None)


def process_library():
    """Show actual parsing/indexing events; do not invent percentages for provider work."""
    with st.status("Checking documents…", expanded=True) as status:
        bar = st.progress(0, text="Files completed")
        def progress(phase, name, complete, total):
            label = {"checking": "Checking documents", "parsing": "Extracting text", "indexing": "Building index", "ready": "Ready"}[phase]
            status.update(label=label + (": " + name if name else ""))
            bar.progress(complete / total if total else 1.0, text=f"{complete} of {total} files completed")
        try:
            store = sync_directory(DATA_DIR, progress=progress)
        except Exception:
            status.update(label="Document processing failed", state="error", expanded=True)
            raise
        status.update(label="Documents ready to ask", state="complete", expanded=False)
        return store


def render_source_reader():
    header, close = st.columns([4, 1])
    with header:
        st.subheader("Source reader")
    with close:
        if st.button("Close", key="close_reader"):
            st.session_state.source_reader_open = False
            st.rerun()
    index = st.session_state.get("source_answer", -1)
    selected = st.session_state.messages[index] if 0 <= index < len(st.session_state.messages) else {}
    sources = selected.get("sources", [])
    current = st.session_state.get("source_id")
    if not sources:
        st.info("This answer has no cited passages.")
        return
    ids = [source["id"] for source in sources]
    if current not in ids:
        current = ids[0]
    if len(sources) > 1:
        current = st.selectbox("Passage", ids, index=ids.index(current),
                               format_func=lambda n: next(f"[{n}] {s['file_name']}" for s in sources if s["id"] == n),
                               key=f"source_select_{index}_{current}")
    source = next(item for item in sources if item["id"] == current)
    st.caption("QUESTION")
    if index > 0:
        st.write(st.session_state.messages[index - 1]["content"])
    label = f"[{source['id']}] {source['file_name']}"
    if source["page"] is not None:
        label += f" · page {source['page']}"
    with st.expander(label, expanded=True):
        st.text(source["excerpt"])
        if source["page_label"] is not None:
            st.caption("Printed page label: " + str(source["page_label"]))
        st.caption("Source: " + source["source"])
    st.caption("These are the exact passages supplied to the answer model. Review them to verify important claims.")


def main():
    stylesheet = Path(__file__).parent / "assets" / "workspace.css"
    st.markdown("<style>" + stylesheet.read_text() + "</style>", unsafe_allow_html=True)
    if "messages" not in st.session_state:
        st.session_state.messages = []
    valid, missing = validate_api_keys()
    notification = st.session_state.pop("notification", None)
    if notification:
        st.toast(notification)

    with st.container(key="workspace_header"):
        title, actions = st.columns([2.4, 1.5])
        with title:
            st.markdown('<div class="workspace-kicker">THE READING ROOM</div>', unsafe_allow_html=True)
            st.title("Document Q&A")
            st.caption("A place for your documents, questions, and discoveries.")
        with actions:
            new, export, settings = st.columns([1.3, 1, 1])
            with new:
                if st.button("New conversation", use_container_width=True):
                    reset_conversation()
                    st.rerun()
            with export:
                st.download_button("Export", conversation_export(st.session_state.messages),
                                   "document-conversation.md", "text/markdown", disabled=not st.session_state.messages,
                                   use_container_width=True)
            with settings:
                with st.popover("Settings", use_container_width=True):
                    st.write("Workspace settings")
                    st.caption(f"Answer model: {LLM_MODEL}\n\nEmbedding model: {EMBEDDING_MODEL}")
                    st.caption("Configure Azure OpenAI credentials and deployment names in .env or Space Settings.")
                    st.caption("Files persist on this app instance. This is a personal or trusted shared library, with no per-user file separation.")
                    st.caption("Scanned PDFs need OCR before import. Follow-up questions should be self-contained.")
    store = None
    sync_error = None
    if valid:
        try:
            store = sync_directory(DATA_DIR)
        except Exception as error:
            sync_error = str(error)

    with st.sidebar:
        st.markdown('<div class="workspace-brand"><span class="workspace-mark" aria-hidden="true">▤</span>Document library</div>', unsafe_allow_html=True)
        st.caption("A collection of ideas, ready to explore.")
        files = sorted(path for path in DATA_DIR.rglob("*") if path.is_file() and path.suffix.lower() in {".pdf", ".txt", ".md"})
        search = st.text_input("Search documents", placeholder="Find a document…", disabled=not files)
        uploads = st.file_uploader("Add documents", type=["pdf", "txt", "md"], accept_multiple_files=True,
                                   help="PDF, TXT or Markdown. Up to 20 MB per file. Same-name files replace the existing version.")
        if st.button("Add to library", type="primary", disabled=not uploads or not valid, use_container_width=True):
            try:
                names = []
                for upload in uploads:
                    names.append(save_upload(upload))
                process_library()
                st.session_state.notification = "Ready: " + ", ".join(names)
                st.rerun()
            except Exception as error:
                st.error(f"Could not process these files: {error}")
        st.divider()
        records = {item["source"]: item for item in list_indexed_documents()} if valid else {}
        st.caption(f"{len(files)} file{'s' if len(files) != 1 else ''} in your library")
        filtered = [path for path in files if search.casefold() in path.name.casefold()]
        for position, path in enumerate(filtered):
            source = str(path.resolve())
            record = records.get(source, {})
            ready = bool(record and record.get("file_hash") == file_digest(path))
            state = "Ready" if ready else "Not indexed"
            with st.expander(path.name + " · " + state):
                st.caption(f"{path.suffix[1:].upper()} · {path.stat().st_size / 1024:.1f} KB · {state}")
                st.caption(str(path.relative_to(DATA_DIR)))
                if not ready and valid:
                    if st.button("Process file", key="process_" + source):
                        try:
                            process_library()
                            st.rerun()
                        except Exception as error:
                            st.error(str(error))
                confirm = st.checkbox("Delete this file and its passages", key="confirm_" + source)
                if st.button("Delete file", key="delete_" + source, disabled=not confirm or not valid):
                    try:
                        path.unlink()
                        sync_directory(DATA_DIR)
                        st.session_state.notification = "Removed " + path.name
                        st.rerun()
                    except Exception as error:
                        st.error(str(error))
        if not files:
            st.info("Your library is empty. Add a document to get started.")
        elif not filtered:
            st.caption("No files match your search.")

    if not valid:
        st.warning("Connect your workspace: add " + ", ".join(missing) + " to .env.")
    if sync_error:
        st.error("A document needs attention before you can ask questions: " + sync_error)
    indexed = list_indexed_documents() if valid else []
    options = {record["source"]: record["file_name"] for record in indexed}
    with st.container(border=True, key="scope_panel"):
        st.caption("ANSWER FROM")
        scope_mode = st.radio("Search scope", ["Entire library", "Selected documents"], horizontal=True, label_visibility="collapsed")
        chosen = None
        if scope_mode == "Selected documents":
            chosen = st.multiselect("Documents to search", list(options), format_func=options.get, placeholder="Choose documents…")
            if not chosen:
                st.info("Choose at least one document to enable questions.")
        if chosen:
            st.caption("In scope: " + ", ".join(options[source] for source in chosen))
        else:
            count = len(indexed) if chosen is None else 0
            st.caption(f"{count} document{'s' if count != 1 else ''} in scope")

    # A nested chat_input renders inline instead of being pinned to the bottom.
    with st.container(key="question_panel"):
        st.markdown('<div class="workspace-kicker">YOUR QUESTION</div>', unsafe_allow_html=True)
        question = st.chat_input("What would you like to discover in your documents?", disabled=store is None or not indexed or chosen == [])

    opened = st.session_state.get("source_reader_open", False)
    if opened:
        conversation, evidence = st.columns([2.1, 1], gap="large")
    else:
        conversation = st.container()
        evidence = None
    pending = None
    with conversation:
        if not st.session_state.messages:
            heading = "Add your first document." if not indexed else "Every document holds a discovery."
            with st.container(key="welcome_panel"):
                st.markdown(
                    '<div class="workspace-welcome">'
                    '<div class="welcome-art" aria-hidden="true"><span></span><i></i><b></b></div>'
                    '<h2>' + heading + '</h2>'
                    '<p>Start with the question box above. Explore an idea, connect the details, and return to the original passage whenever you need.</p>'
                    '<div class="workspace-steps"><span><b>1</b>Add documents</span><span><b>2</b>Ask a question</span><span><b>3</b>Check sources</span></div></div>',
                    unsafe_allow_html=True,
                )
                if indexed:
                    st.caption("A starting point — adapt these questions to your documents")
                    prompts = ["What do the documents say about retrieval?", "What limitations are discussed in the documents?"]
                    cards = st.columns(2)
                    for card, prompt in zip(cards, prompts):
                        with card:
                            if st.button(prompt, disabled=store is None or chosen == [], use_container_width=True):
                                pending = prompt

        for index, message in enumerate(st.session_state.messages):
            with st.chat_message(message["role"]):
                if message.get("status") == "insufficient_evidence":
                    st.warning(message["content"])
                elif message.get("status") == "error":
                    st.error(message["content"])
                else:
                    st.markdown(message["content"])
                if message.get("sources"):
                    st.caption("Open a source passage")
                    for source in message["sources"]:
                        label = f"[{source['id']}] {source['file_name']}"
                        if source["page"] is not None:
                            label += f" · p. {source['page']}"
                        if st.button(label, key=f"cite_{index}_{source['id']}"):
                            st.session_state.source_answer = index
                            st.session_state.source_id = source["id"]
                            st.session_state.source_reader_open = True
                            st.rerun()
        if evidence is not None:
            with evidence:
                with st.container(key="source_reader"):
                    render_source_reader()
    if pending:
        question = pending
    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        try:
            chain = create_rag_chain(create_retriever(store, sources=chosen))
            with st.spinner("Finding supporting passages…"):
                result = chain.invoke(question)
            st.session_state.messages.append({"role": "assistant", "content": result["answer"], "sources": result["sources"], "status": result["status"]})
        except Exception as error:
            st.session_state.messages.append({"role": "assistant", "content": f"Could not complete this question: {error}", "status": "error"})
        st.rerun()


if __name__ == "__main__":
    main()
