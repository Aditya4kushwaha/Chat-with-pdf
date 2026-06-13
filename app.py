import os
import json
from pathlib import Path

import google.generativeai as genai
from pypdf import PdfReader
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# ── Config ─────────────────────────────────────────────────────────────────────
CHUNK_SIZE    = 1000  # Character count for RecursiveCharacterTextSplitter
CHUNK_OVERLAP = 150   # Character overlap
TOP_K         = 4
UPLOAD_DIR    = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    raise RuntimeError("GEMINI_API_KEY environment variable not set.")

genai.configure(api_key=api_key)
model = genai.GenerativeModel("gemini-2.5-flash")

# Initialize LangChain Embeddings & Vector Store
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001", google_api_key=api_key)
vector_store = InMemoryVectorStore(embeddings)

app = FastAPI(title="RAG Chatbot — Gemini")
app.mount("/static", StaticFiles(directory="static"), name="static")

# ── In-memory document store ───────────────────────────────────────────────────
# Maps filename -> list of chunk IDs in vector_store
document_chunks: dict[str, list[str]] = {}


# ── RAG utilities ──────────────────────────────────────────────────────────────
def chunk_text(text: str, source: str) -> list[Document]:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    return text_splitter.create_documents(
        texts=[text],
        metadatas=[{"source": source}]
    )


def extract_text_from_file(file_path: Path) -> str:
    ext = file_path.suffix.lower()
    if ext == ".pdf":
        try:
            reader = PdfReader(file_path)
            text_parts = []
            for page in reader.pages:
                text_parts.append(page.extract_text() or "")
            return "\n".join(text_parts)
        except Exception as e:
            raise RuntimeError(f"Failed to read PDF: {e}")
    else:
        # text files (.txt, .md, .csv)
        try:
            return file_path.read_text(encoding="utf-8", errors="replace")
        except Exception as e:
            raise RuntimeError(f"Failed to read text file: {e}")


def retrieve(query: str) -> list[Document]:
    if not any(document_chunks.values()):
        return []
    return vector_store.similarity_search(query, k=TOP_K)


# ── Pydantic models ────────────────────────────────────────────────────────────
class ChatRequest(BaseModel):
    question: str

class ChatResponse(BaseModel):
    answer:      str
    sources:     list[str]
    chunks_used: int


# ── API routes ─────────────────────────────────────────────────────────────────
@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    allowed = {".txt", ".md", ".csv", ".pdf"}
    ext     = Path(file.filename).suffix.lower()
    if ext not in allowed:
        raise HTTPException(400, f"Unsupported type '{ext}'. Allowed: {allowed}")
    content = await file.read()
    target_path = UPLOAD_DIR / file.filename
    target_path.write_bytes(content)
    try:
        text = extract_text_from_file(target_path)
    except Exception as e:
        if target_path.exists():
            target_path.unlink()
        raise HTTPException(400, str(e))
    
    # Chunk text using RecursiveCharacterTextSplitter
    chunks = chunk_text(text, file.filename)
    doc_ids = [f"{file.filename}::{i}" for i in range(len(chunks))]
    
    # If the file already exists in the vector store, delete its old chunks first
    if file.filename in document_chunks:
        try:
            vector_store.delete(ids=document_chunks[file.filename])
        except Exception as e:
            print(f"Failed to delete old chunks for {file.filename}: {e}")

    # Add to vector store and update tracking
    vector_store.add_documents(documents=chunks, ids=doc_ids)
    document_chunks[file.filename] = doc_ids

    return {"filename": file.filename, "chunks": len(chunks), "words": len(text.split())}


@app.delete("/document/{filename}")
async def delete_doc(filename: str):
    if filename not in document_chunks:
        raise HTTPException(404, "Document not found")
    
    try:
        vector_store.delete(ids=document_chunks[filename])
    except Exception as e:
        raise HTTPException(500, f"Failed to delete chunks from vector store: {e}")

    del document_chunks[filename]
    p = UPLOAD_DIR / filename
    if p.exists():
        p.unlink()
    return {"deleted": filename}


@app.get("/documents")
async def list_documents():
    return [{"filename": n, "chunks": len(cs)} for n, cs in document_chunks.items()]


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    retrieved = retrieve(req.question)

    if retrieved:
        context = "\n\n".join(
            f"[{i+1}] (source: {c.metadata.get('source', 'Unknown')})\n{c.page_content}"
            for i, c in enumerate(retrieved)
        )
    else:
        context = "No relevant documents found in the knowledge base."

    prompt = (
        "You are a helpful assistant. Answer the user's question using ONLY the "
        "context provided below. If the answer is not found in the context, say so clearly.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {req.question}"
    )

    response = model.generate_content(prompt)
    answer   = response.text
    sources  = list({c.metadata.get("source", "Unknown") for c in retrieved})
    return ChatResponse(answer=answer, sources=sources, chunks_used=len(retrieved))


@app.on_event("startup")
def startup_event():
    if UPLOAD_DIR.exists():
        for file_path in UPLOAD_DIR.iterdir():
            if file_path.is_file() and not file_path.name.startswith("."):
                allowed = {".txt", ".md", ".csv", ".pdf"}
                if file_path.suffix.lower() in allowed:
                    try:
                        text = extract_text_from_file(file_path)
                        chunks = chunk_text(text, file_path.name)
                        doc_ids = [f"{file_path.name}::{i}" for i in range(len(chunks))]
                        vector_store.add_documents(documents=chunks, ids=doc_ids)
                        document_chunks[file_path.name] = doc_ids
                        print(f"Indexed startup file: {file_path.name} ({len(chunks)} chunks)")
                    except Exception as e:
                        print(f"Failed to index startup file {file_path.name}: {e}")

# ── Serve frontend ─────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def index():
    p = Path("static/index.html")
    return p.read_text() if p.exists() else "<h1>RAG Chatbot (Gemini)</h1>"
