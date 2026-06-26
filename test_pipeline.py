"""
test_pipeline.py
Quick test to verify each component works before running the full server.

Run with:
    python test_pipeline.py

You need:
  - pip install requirements.txt done
  - .env file with GROQ_API_KEY set
"""
import os
import sys
from pathlib import Path

# Allow imports from parent directory
sys.path.insert(0, str(Path(__file__).parent))

print("=" * 55)
print("   Resume Chatbot — Component Tests")
print("=" * 55)


# ─── Test 1: Config ───────────────────────────────────────
print("\n[1] Testing config...")
try:
    from app.config import get_settings
    settings = get_settings()
    groq_ok = bool(settings.groq_api_key and settings.groq_api_key != "your_groq_api_key_here")
    print(f"    GROQ_API_KEY set:  {'✅ YES' if groq_ok else '❌ NO — add it to .env'}")
    print(f"    Embedding model:   {settings.embedding_model}")
    print(f"    Chroma path:       {settings.chroma_path}")
except Exception as e:
    print(f"    ❌ Config error: {e}")
    sys.exit(1)


# ─── Test 2: PDF Parser ───────────────────────────────────
print("\n[2] Testing PDF parser (chunk logic only)...")
try:
    from app.pdf_parser import chunk_text
    sample = "Hello world " * 300  # 600 words
    chunks = chunk_text(sample, chunk_size=100, overlap=10)
    print(f"    Input words:   600")
    print(f"    Chunks made:   {len(chunks)} ✅")
    print(f"    First chunk:   {chunks[0][:60]}...")
except Exception as e:
    print(f"    ❌ Parser error: {e}")


# ─── Test 3: Embedding model ──────────────────────────────
print("\n[3] Testing sentence-transformers embedding...")
try:
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(settings.embedding_model)
    vec = model.encode(["Python developer with 3 years experience"])
    print(f"    Model loaded:    ✅")
    print(f"    Vector dims:     {len(vec[0])} (expected 384)")
except Exception as e:
    print(f"    ❌ Embedding error: {e}")
    print(f"       Run: pip install sentence-transformers")


# ─── Test 4: ChromaDB ─────────────────────────────────────
print("\n[4] Testing ChromaDB vector store...")
try:
    import chromadb
    Path("./data/chroma_test").mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path="./data/chroma_test")
    col = client.get_or_create_collection("test")
    col.add(ids=["t1"], documents=["test doc"], embeddings=[[0.1] * 384])
    result = col.query(query_embeddings=[[0.1] * 384], n_results=1)
    client.delete_collection("test")
    print(f"    ChromaDB:         ✅ working")
except Exception as e:
    print(f"    ❌ ChromaDB error: {e}")
    print(f"       Run: pip install chromadb")


# ─── Test 5: Groq API ─────────────────────────────────────
print("\n[5] Testing Groq API...")
if not groq_ok:
    print("    ⏭️  Skipped — GROQ_API_KEY not set")
else:
    try:
        from groq import Groq
        client = Groq(api_key=settings.groq_api_key)
        resp = client.chat.completions.create(
            model="llama-3.1-8b-instant",  # Updated from decommissioned model
            messages=[{"role": "user", "content": "Say: API test OK"}],
            max_tokens=20
        )
        reply = resp.choices[0].message.content
        print(f"    Groq response:   ✅ '{reply.strip()}'")
    except Exception as e:
        print(f"    ❌ Groq error: {e}")
        print(f"       Check your GROQ_API_KEY in .env")


# ─── Summary ──────────────────────────────────────────────
print("\n" + "=" * 55)
print("  Done! If all ✅, run:  uvicorn main:app --reload")
print("  Then open:             http://localhost:8000/docs")
print("=" * 55 + "\n")