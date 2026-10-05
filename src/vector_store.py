"""Persistent, incremental Chroma indexing. Existing vectors are never deleted on startup."""
import hashlib
import json
import threading
from collections import defaultdict
from pathlib import Path
from langchain_chroma import Chroma
from config import DB_DIR, COLLECTION_NAME, EMBEDDING_IDENTITY, CHUNK_SIZE, CHUNK_OVERLAP
from src.embeddings import get_embeddings

INDEX_VERSION = 2
SIGNATURE = hashlib.sha256(json.dumps([INDEX_VERSION, EMBEDDING_IDENTITY, CHUNK_SIZE, CHUNK_OVERLAP]).encode()).hexdigest()[:12]
ACTIVE_COLLECTION = f"{COLLECTION_NAME}_{SIGNATURE}"
_LOCK = threading.RLock()


def _manifest_path():
    return DB_DIR / f"{ACTIVE_COLLECTION}.json"


def _read_manifest():
    path = _manifest_path()
    return json.loads(path.read_text()) if path.exists() else {}


def _write_manifest(manifest):
    path = _manifest_path()
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    temp.replace(path)


def open_vector_store(embedding=None):
    return Chroma(collection_name=ACTIVE_COLLECTION,
                  embedding_function=embedding if embedding is not None else get_embeddings(),
                  persist_directory=str(DB_DIR), collection_metadata={"hnsw:space": "cosine"})


def _chunk_id(doc):
    data = [doc.metadata.get("source"), doc.metadata.get("page"), doc.metadata.get("start_index"), doc.page_content]
    return hashlib.sha256(json.dumps(data, ensure_ascii=False).encode()).hexdigest()


def create_vector_store(documents, clear_existing=False, vector_store=None):
    """Upsert only changed sources. Unchanged chunks incur no embedding calls.

    clear_existing is retained for compatibility but refused: use explicit per-source deletion.
    """
    if clear_existing:
        raise ValueError("Destructive rebuild disabled; delete individual documents explicitly")
    if not documents:
        raise ValueError("No documents provided")
    store = vector_store if vector_store is not None else open_vector_store()
    grouped = defaultdict(list)
    for doc in documents:
        if doc.page_content.strip():
            grouped[doc.metadata["source"]].append(doc)
    if not grouped:
        raise ValueError("No extractable text. Scanned PDFs need OCR before indexing.")
    with _LOCK:
        manifest = _read_manifest()
        for source, chunks in grouped.items():
            # Collapse identical chunks, preserving page/offset distinctions.
            unique = {_chunk_id(doc): doc for doc in chunks}
            ids = list(unique)
            old = manifest.get(source, {})
            existing = set(store.get(ids=ids, include=[])['ids'])
            missing = [chunk_id for chunk_id in ids if chunk_id not in existing]
            for offset in range(0, len(missing), 50):
                batch = missing[offset:offset + 50]
                store.add_documents([unique[chunk_id] for chunk_id in batch], ids=batch)
            # Only retire old vectors after all replacement vectors have been written.
            source_ids = store.get(where={"source": source}, include=[])["ids"]
            stale = list(set(source_ids) - set(ids))
            if stale:
                store.delete(ids=stale)
            manifest[source] = {"ids": ids, "file_name": chunks[0].metadata.get("file_name", Path(source).name)}
        _write_manifest(manifest)
    return store


def load_vector_store():
    store = open_vector_store()
    return store if store._collection.count() else None


def get_or_create_vector_store(documents=None):
    if documents:
        return create_vector_store(documents)
    store = load_vector_store()
    if store is None:
        raise ValueError("No indexed documents. Add files and run --index.")
    return store


def list_indexed_documents():
    with _LOCK:
        return [{"source": source, **record} for source, record in _read_manifest().items()]


def delete_indexed_document(source, vector_store=None):
    store = vector_store if vector_store is not None else open_vector_store()
    with _LOCK:
        manifest = _read_manifest()
        record = manifest.get(source)
        if record:
            store.delete(where={"source": source})
            del manifest[source]
            _write_manifest(manifest)


def file_digest(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sync_directory(directory, vector_store=None, progress=None):
    """Parse/embed new or changed files only; remove entries for deleted local files."""
    from src.document_loader import load_single_document
    from src.text_splitter import split_documents
    store = vector_store if vector_store is not None else open_vector_store()
    directory = Path(directory).resolve()
    files = sorted(path for path in directory.rglob("*") if path.is_file() and path.suffix.lower() in {".pdf", ".txt", ".md"})
    current = {str(path.resolve()) for path in files}
    with _LOCK:
        manifest = _read_manifest()
        total = len(files)
        if progress:
            progress("checking", "", 0, total)
        for position, path in enumerate(files):
            source = str(path.resolve())
            digest = file_digest(path)
            record = manifest.get(source, {})
            if record.get("file_hash") == digest:
                ids = record.get("ids", [])
                if ids and len(store.get(ids=ids, include=[])['ids']) == len(ids):
                    if progress:
                        progress("ready", path.name, position + 1, total)
                    continue
            if progress:
                progress("parsing", path.name, position, total)
            docs = load_single_document(path.resolve())
            if not docs or not any(doc.page_content.strip() for doc in docs):
                raise ValueError(f"Could not extract text from {path.name}. Check the file; scanned PDFs need OCR.")
            chunks = split_documents(docs)
            if progress:
                progress("indexing", path.name, position, total)
            create_vector_store(chunks, vector_store=store)
            manifest = _read_manifest()
            manifest[source]["file_hash"] = digest
            _write_manifest(manifest)
            if progress:
                progress("ready", path.name, position + 1, total)
        for source in list(manifest):
            if source.startswith(("http://", "https://")):
                continue
            path = Path(source).resolve()
            if path.is_relative_to(directory) and source not in current:
                delete_indexed_document(source, vector_store=store)
    return store
