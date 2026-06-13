# RAG Chatbot — Python + FastAPI + Google Gemini

## Quick Start

### 1. Get a Gemini API key
Go to https://aistudio.google.com/app/apikey and create a free API key.

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

### 4. Set your Gemini API key

**Mac/Linux:**
```bash
export GEMINI_API_KEY=your-key-here
```

**Windows CMD:**
```cmd
set GEMINI_API_KEY=your-key-here
```

**Windows PowerShell:**
```powershell
$env:GEMINI_API_KEY="your-key-here"
```

### 5. Run
```bash
python run.py
```

Open **http://localhost:8000**

---

## Project Structure
```
rag-gemini/
├── app.py            # FastAPI + Gemini backend
├── run.py            # Server launcher
├── requirements.txt  # Dependencies
├── static/
│   └── index.html    # Chat UI
└── uploads/          # Saved documents
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | Chat UI |
| POST | `/upload` | Upload a .txt/.md/.csv file |
| GET | `/documents` | List indexed documents |
| DELETE | `/document/{filename}` | Remove a document |
| POST | `/chat` | Ask a question |

## Upgrade Ideas
- Real embeddings: `google-generativeai` has `embed_content()` for semantic search
- PDF support: add `pypdf`
- Persistent storage: `chromadb` or `sqlite`
- Streaming: use Gemini streaming API + SSE
