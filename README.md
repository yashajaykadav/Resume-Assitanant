# AI Resume Chatbot — Backend

RAG-powered chatbot that answers questions about any PDF resume.
**100% free stack**: Groq API + ChromaDB + sentence-transformers + FastAPI.

---

## Quick Setup (5 steps)

### 1. Clone / enter project folder
```bash
cd resume_chatbot
```

### 2. Create virtual environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac / Linux
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```
> First run downloads the embedding model (~80MB). After that it's cached.

### 4. Set your free Groq API key
```bash
cp .env.example .env
```
- Go to https://console.groq.com → sign up free → create an API key
- Paste it into `.env`:
```
GROQ_API_KEY=gsk_xxxxxxxxxxxxxxxxxxxx
```

### 5. Run the server
```bash
uvicorn main:app --reload --port 8000
```

Open **http://localhost:8000/docs** — you'll see the full interactive API.

---

## Testing

Run the component test before the server to verify everything works:
```bash
python test_pipeline.py
```

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/v1/health` | Server health check |
| `POST` | `/api/v1/upload` | Upload a PDF resume |
| `POST` | `/api/v1/chat` | Ask a question |
| `GET` | `/api/v1/status/{user_id}` | Check resume status |
| `DELETE` | `/api/v1/resume/{user_id}` | Delete resume data |

---

## Testing with curl

### Upload a resume
```bash
curl -X POST http://localhost:8000/api/v1/upload \
  -F "file=@your_resume.pdf" \
  -F "user_id=testuser"
```

### Ask a question
```bash
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "testuser",
    "question": "What programming languages does the candidate know?",
    "chat_history": []
  }'
```

### Multi-turn conversation
```bash
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "testuser",
    "question": "Which of those is their strongest skill?",
    "chat_history": [
      {"role": "user", "content": "What languages do they know?"},
      {"role": "assistant", "content": "Python, Java, and JavaScript."}
    ]
  }'
```

---

## Project Structure

```
resume_chatbot/
├── main.py              ← FastAPI app, startup, CORS
├── requirements.txt     ← All dependencies
├── .env.example         ← Copy to .env and add your keys
├── test_pipeline.py     ← Component tests (run before server)
│
├── app/
│   ├── config.py        ← Settings from .env
│   ├── pdf_parser.py    ← PDF → text chunks (PyMuPDF)
│   ├── vector_store.py  ← Embeddings + ChromaDB
│   ├── rag_engine.py    ← Groq LLM calls
│   ├── routes.py        ← All API endpoints
│   └── schemas.py       ← Pydantic request/response models
│
├── uploads/             ← Saved PDF files (auto-created)
└── data/
    └── chroma/          ← ChromaDB vector storage (auto-created)
```

---

## Free Tier Limits

| Service | Free Limit |
|---------|-----------|
| Groq API | 14,400 requests/day, 30 req/min |
| sentence-transformers | Unlimited (runs locally) |
| ChromaDB | Unlimited (runs locally) |
| Render.com (deploy) | 750 hours/month |

---

## Next Steps

- [ ] Add JWT authentication (FastAPI-users)
- [ ] Add Redis for rate limiting
- [ ] Connect Flutter app to `/upload` and `/chat`
- [ ] Connect React web app to the same endpoints
- [ ] Deploy backend to Render.com (free)
