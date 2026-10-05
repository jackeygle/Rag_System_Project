"""
Document Loader Module
Supports PDF, TXT, MD, and Web pages
"""
import hashlib
from datetime import datetime
from pathlib import Path
from typing import List, Optional
from langchain_core.documents import Document
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    WebBaseLoader,
)
from tqdm import tqdm


def generate_doc_id(content: str, source: str) -> str:
    """Generate a unique document ID based on content hash."""
    combined = f"{source}:{content[:500]}"
    return hashlib.md5(combined.encode()).hexdigest()[:16]


def load_single_document(file_path: Path) -> List[Document]:
    """Load a single document based on its file extension."""
    file_path = file_path.resolve()
    suffix = file_path.suffix.lower()
    
    try:
        if suffix == ".pdf":
            loader = PyPDFLoader(str(file_path))
        elif suffix in [".txt", ".md"]:
            # Use TextLoader for both txt and md files
            loader = TextLoader(str(file_path), encoding="utf-8")
        else:
            print(f"⚠️  Unsupported file type: {suffix} ({file_path.name})")
            return []
        
        docs = loader.load()
        
        # Add enriched metadata
        timestamp = datetime.now().isoformat()
        for doc in docs:
            doc.metadata["source"] = str(file_path)
            doc.metadata["file_name"] = file_path.name
            doc.metadata["file_type"] = suffix[1:]  # Remove the dot
            doc.metadata["loaded_at"] = timestamp
            doc.metadata["doc_id"] = generate_doc_id(doc.page_content, str(file_path))
        return docs
    except UnicodeDecodeError:
        # Try with different encoding
        try:
            loader = TextLoader(str(file_path), encoding="latin-1")
            docs = loader.load()
            for doc in docs:
                doc.metadata["source"] = str(file_path)
                doc.metadata["file_name"] = file_path.name
                doc.metadata["encoding"] = "latin-1"
            return docs
        except Exception as e:
            print(f"❌ Error loading {file_path.name}: {e}")
            return []
    except Exception as e:
        print(f"❌ Error loading {file_path.name}: {e}")
        return []


def load_documents_from_directory(directory: Path) -> List[Document]:
    """Load all supported documents from a directory."""
    all_documents = []
    supported_extensions = {".pdf", ".txt", ".md"}
    
    # Find all supported files
    files = [
        f for f in directory.rglob("*") 
        if f.is_file() and f.suffix.lower() in supported_extensions
    ]
    
    if not files:
        print(f"📂 No supported documents found in {directory}")
        return []
    
    print(f"📂 Found {len(files)} document(s) to load...")
    
    for file_path in tqdm(files, desc="Loading documents"):
        docs = load_single_document(file_path)
        all_documents.extend(docs)
    
    print(f"✅ Loaded {len(all_documents)} document chunk(s)")
    return all_documents


def load_from_urls(urls: List[str]) -> List[Document]:
    """Load documents from web URLs."""
    if not urls:
        return []
    
    print(f"🌐 Loading {len(urls)} web page(s)...")
    
    try:
        loader = WebBaseLoader(urls)
        docs = loader.load()
        
        # Add enriched metadata
        timestamp = datetime.now().isoformat()
        for doc, url in zip(docs, urls):
            doc.metadata["source"] = url
            doc.metadata["type"] = "web"
            doc.metadata["loaded_at"] = timestamp
            doc.metadata["doc_id"] = generate_doc_id(doc.page_content, url)
        
        print(f"✅ Loaded {len(docs)} web page(s)")
        return docs
    except Exception as e:
        print(f"❌ Error loading URLs: {e}")
        return []


def load_all_documents(
    directory: Optional[Path] = None,
    urls: Optional[List[str]] = None
) -> List[Document]:
    """Load documents from both directory and URLs."""
    all_docs = []
    
    if directory:
        all_docs.extend(load_documents_from_directory(directory))
    
    if urls:
        all_docs.extend(load_from_urls(urls))
    
    return all_docs

