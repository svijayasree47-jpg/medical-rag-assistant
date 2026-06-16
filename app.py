from flask import Flask, request, jsonify, render_template
import os
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import chromadb
import ollama

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load embedding model
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Setup ChromaDB
chroma_client = chromadb.PersistentClient(path="database")
collection = chroma_client.get_or_create_collection(name="medical_docs")


def extract_text_from_pdf(filepath):
    reader = PdfReader(filepath)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text


def split_into_chunks(text, chunk_size=500):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
    return chunks


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():
    file = request.files.get("file")
    if not file or not file.filename.endswith(".pdf"):
        return jsonify({"error": "Please upload a PDF file"}), 400

    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)

    # Extract and chunk
    text = extract_text_from_pdf(filepath)
    chunks = split_into_chunks(text)

    # Store in ChromaDB
    for i, chunk in enumerate(chunks):
        doc_id = f"{file.filename}_{i}"
        embedding = embedder.encode(chunk).tolist()
        collection.upsert(
            ids=[doc_id],
            documents=[chunk],
            embeddings=[embedding]
        )

    return jsonify({"message": f"Uploaded and processed '{file.filename}' ({len(chunks)} chunks)"})


@app.route("/ask", methods=["POST"])
def ask():
    data = request.json
    question = data.get("question", "").strip()
    if not question:
        return jsonify({"error": "No question provided"}), 400

    # Embed question and search
    q_embedding = embedder.encode(question).tolist()
    results = collection.query(query_embeddings=[q_embedding], n_results=3)
    context_chunks = results["documents"][0]
    context = "\n\n".join(context_chunks)

    # Ask Llama 3 via Ollama
    prompt = f"""You are a helpful medical information assistant.
Answer the question using only the context below from uploaded medical documents.
If the answer is not in the context, say "I could not find this in the uploaded documents."

Context:
{context}

Question: {question}

Answer:"""

    response = ollama.chat(
        model="llama3",
        messages=[{"role": "user", "content": prompt}]
    )
    answer = response["message"]["content"]

    return jsonify({"answer": answer})


if __name__ == "__main__":
    app.run(debug=True)

