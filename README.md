# RAG Chatbot — LangChain + FastAPI + Google Gemini

A Retrieval-Augmented Generation (RAG) chatbot web application that allows users to upload documents (PDF, TXT, MD, CSV) and chat with them using semantic vector search powered by **LangChain** and **Google Gemini**.

## Architecture & Features

This application implements a complete modern RAG pipeline using:
- **LangChain Text Splitters**: Uses `RecursiveCharacterTextSplitter` to split documents into semantically coherent chunks by looking at paragraphs, sentences, and words.
- **LangChain Google GenAI Embeddings**: Uses `GoogleGenerativeAIEmbeddings` with `models/gemini-embedding-001` to generate vector embeddings for each document chunk.
- **LangChain In-Memory Vector Store**: Uses `InMemoryVectorStore` to index, query, and manage document chunk embeddings locally in memory.
- **Google Gemini LLM**: Grounds chatbot answers using `gemini-2.5-flash` with context retrieved from the vector store.
- **FastAPI Backend**: Simple and efficient REST API to handle uploads, list documents, delete documents, and chat.
- **Modern UI**: Dark mode web interface with drag-and-drop file uploading and micro-animations.

---

## Quick Start

### 1. Get a Gemini API key
Go to [Google AI Studio](https://aistudio.google.com/app/apikey) and create a free API key.

### 2. Create virtual environment
```bash
python -m venv venv

# Mac/Linux
source venv/bin/activate

# Windows
venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables
Create a `.env` file in the project root:
```env
GEMINI_API_KEY="your-api-key-here"
```

### 5. Run the Server
```bash
python run.py
```

Open **http://localhost:8000** in your browser.

---

## Project Structure
```
rag-gemini/
├── app.py            # FastAPI backend (LangChain pipeline & endpoints)
├── run.py            # Server launcher script
├── requirements.txt  # Project dependencies
├── static/
│   └── index.html    # Single-page Frontend Chat UI
└── uploads/          # Directory where uploaded documents are stored
```

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | Serves the Chat UI |
| POST | `/upload` | Upload and index a file (.txt, .md, .csv, .pdf) |
| GET | `/documents` | List currently indexed documents and chunk counts |
| DELETE | `/document/{filename}` | Delete a document and its chunks from the vector store |
| POST | `/chat` | Chat query to retrieve context and generate grounded answer |

---

## Upgrade Ideas
- **Persistent Vector Store**: Upgrade `InMemoryVectorStore` to a persistent vector store like `Chroma` or `FAISS` to keep embeddings between restarts.
- **Metadata Filtering**: Leverage LangChain metadata filters to query specific files during search.
- **Streaming Responses**: Integrate FastAPI's `StreamingResponse` with LangChain `astream` to stream model answers in real-time.
