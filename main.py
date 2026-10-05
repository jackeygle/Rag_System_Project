#!/usr/bin/env python3
"""
RAG System - Main Entry Point
A Retrieval-Augmented Generation system using Groq LLM
"""
import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from config import DATA_DIR, DB_DIR
from src.document_loader import load_all_documents
from src.text_splitter import split_documents
from src.vector_store import load_vector_store, create_vector_store, sync_directory
from src.retriever import create_retriever
from src.generator import create_rag_chain, query


def index_documents(urls: list = None):
    """Index documents from the data directory and optional URLs."""
    print("\n" + "="*50)
    print("📚 RAG System - Document Indexing")
    print("="*50 + "\n")
    
    # Reuse unchanged files; web pages are explicitly refreshed when requested.
    store = sync_directory(DATA_DIR)
    if not urls and not store._collection.count():
        print("No readable documents found. Add PDF, TXT or MD files.")
        return False
    if not urls:
        print("✅ Local document index synchronized.")
        return True
    documents = load_all_documents(urls=urls)
    
    if not documents:
        print("\n❌ No documents found to index!")
        print(f"   Please add documents to: {DATA_DIR}")
        print("   Supported formats: PDF, TXT, MD")
        return False
    
    # Split documents
    chunks = split_documents(documents)
    
    if not chunks:
        print("\n❌ Failed to split documents!")
        return False
    
    # Create vector store
    create_vector_store(chunks)
    
    print("\n✅ Indexing complete!")
    return True


def interactive_query():
    """Start interactive query mode."""
    print("\n" + "="*50)
    print("🤖 RAG System - Interactive Mode")
    print("="*50)
    print("\nCommands:")
    print("  - Type your question and press Enter")
    print("  - Type 'quit' or 'exit' to exit")
    print("  - Type 'clear' to clear screen")
    print("-"*50 + "\n")
    
    # Load vector store
    vector_store = load_vector_store()
    
    if vector_store is None:
        print("❌ No indexed documents found!")
        print(f"   Please add documents to: {DATA_DIR}")
        print("   Then run: python main.py --index")
        return
    
    # Create RAG chain
    retriever = create_retriever(vector_store)
    rag_chain = create_rag_chain(retriever)
    
    print("✅ RAG system ready! Ask me anything about your documents.\n")
    
    while True:
        try:
            question = input("📝 You: ").strip()
            
            if not question:
                continue
            
            if question.lower() in ["quit", "exit", "q"]:
                print("\n👋 Goodbye!")
                break
            
            if question.lower() == "clear":
                print("\033[H\033[J")  # Clear screen
                continue
            
            print("\n🔍 Searching documents...")
            response = query(rag_chain, question)
            print(f"\n🤖 Assistant:\n{response}\n")
            print("-"*50 + "\n")
            
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}\n")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="RAG System - Retrieval Augmented Generation with Groq",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --index              Index documents in data/documents/
  python main.py --index --url URL    Index documents + web page
  python main.py                      Start interactive query mode
  python main.py --query "question"   Single query mode
        """
    )
    
    parser.add_argument(
        "--index", "-i",
        action="store_true",
        help="Index documents from data/documents/ directory"
    )
    
    parser.add_argument(
        "--url", "-u",
        action="append",
        help="URL to index (can be used multiple times)"
    )
    
    parser.add_argument(
        "--query", "-q",
        type=str,
        help="Single query mode (non-interactive)"
    )
    
    args = parser.parse_args()
    
    # Index mode
    if args.index:
        success = index_documents(urls=args.url)
        if not success:
            sys.exit(1)
        
        # If no query specified, exit after indexing
        if not args.query:
            return
    
    # Single query mode
    if args.query:
        vector_store = load_vector_store()
        if vector_store is None:
            print("❌ No indexed documents. Run with --index first.")
            sys.exit(1)
        
        retriever = create_retriever(vector_store)
        rag_chain = create_rag_chain(retriever)
        
        print(f"\n📝 Question: {args.query}\n")
        response = query(rag_chain, args.query)
        print(f"🤖 Answer:\n{response}\n")
        return
    
    # Interactive mode (default)
    interactive_query()


if __name__ == "__main__":
    main()
