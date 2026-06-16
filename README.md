# Medical Knowledge Assistant (RAG)

A Retrieval-Augmented Generation (RAG) based Medical Knowledge Assistant built with Flask and LLMs.

## What it does
- Upload medical PDF documents
- Ask questions in natural language
- AI answers only from the uploaded documents

## Tech Stack
- Python & Flask
- Sentence Transformers
- ChromaDB (Vector Database)
- Llama 3 (via Ollama)
- PyPDF

## How to Run
1. Install Ollama and pull Llama 3
2. Install dependencies: `pip install flask pypdf sentence-transformers chromadb ollama`
3. Run: `python app.py`
4. Open: `http://localhost:5000`
