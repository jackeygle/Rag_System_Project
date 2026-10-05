#!/usr/bin/env python3
"""
RAG System - Gradio Web Interface (Simplified for HF Spaces)
"""
import os
import sys
import gradio as gr
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv
load_dotenv()

from config import DATA_DIR, validate_api_keys
from src.document_loader import load_all_documents
from src.text_splitter import split_documents
from src.vector_store import create_vector_store, sync_directory
from src.retriever import create_retriever
from src.generator import create_rag_chain, query

# Global variables
rag_chain = None
init_status = "⏳ Initializing..."


def initialize_rag():
    """Initialize the RAG system on startup."""
    global rag_chain, init_status
    
    valid, missing = validate_api_keys()
    if not valid:
        init_status = "Missing configuration: " + ", ".join(missing)
        return False

    print("✅ API Keys configured")
    
    try:
        vector_store = sync_directory(DATA_DIR)
        chunk_count = vector_store._collection.count()
        if not chunk_count:
            init_status = "No indexed documents. Add files to data/documents/."
            return False

        retriever = create_retriever(vector_store)
        rag_chain = create_rag_chain(retriever)
        
        init_status = f"✅ Loaded {chunk_count} document chunks"
        print("✅ RAG system initialized successfully!")
        return True
        
    except Exception as e:
        import traceback
        init_status = f"❌ Error: {str(e)[:50]}"
        print(f"❌ Initialization error: {e}")
        traceback.print_exc()
        return False


def respond(message, history):
    """Handle chat messages."""
    global rag_chain
    
    if not message.strip():
        return history
    
    if rag_chain is None:
        history.append([message, "⚠️ System not initialized. Please check API key configuration."])
        return history
    
    try:
        response = query(rag_chain, message)
        history.append([message, response])
    except Exception as e:
        history.append([message, f"❌ 错误: {str(e)}"])
    
    return history


# Initialize on startup
print("🚀 Initializing RAG system...")
initialize_rag()

# Create simple Gradio interface
with gr.Blocks(
    title="RAG 智能文档助手",
    theme=gr.themes.Soft(primary_hue="violet"),
) as demo:
    
    gr.Markdown("""
    # 🤖 RAG Document Assistant
    AI-powered document Q&A using Retrieval-Augmented Generation
    """)
    
    gr.Markdown(f"**Status**: {init_status}")
    
    chatbot = gr.Chatbot(height=450, label="Chat")
    
    with gr.Row():
        msg = gr.Textbox(
            placeholder="💬 Enter your question, press Enter to send...",
            show_label=False,
            scale=9,
        )
        send = gr.Button("Send", scale=1, variant="primary")
    
    with gr.Row():
        gr.Button("What is machine learning?").click(
            lambda h: respond("What is machine learning?", h), [chatbot], [chatbot]
        )
        gr.Button("What are RAG advantages?").click(
            lambda h: respond("What are the advantages of RAG?", h), [chatbot], [chatbot]
        )
        gr.Button("How does gradient descent work?").click(
            lambda h: respond("How does gradient descent work?", h), [chatbot], [chatbot]
        )
    
    gr.Markdown("---\n*Powered by LangChain + Azure OpenAI + ChromaDB*")
    
    msg.submit(respond, [msg, chatbot], [chatbot]).then(lambda: "", None, [msg])
    send.click(respond, [msg, chatbot], [chatbot]).then(lambda: "", None, [msg])


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
