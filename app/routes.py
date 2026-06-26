"""
app/routes.py
All API endpoints for the Resume Chatbot.

Endpoints:
  POST /upload           → upload a PDF resume
  POST /chat             → ask a question about the resume
  POST /chat/stream      → stream AI response token-by-token
  GET  /status/{user_id} → check if a resume is uploaded
  GET  /chat-history/{user_id} → fetch chat logs
  DELETE /resume/{user_id} → delete a user's resume
  GET  /health           → server health check
"""
import os
import uuid
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import List

from fastapi import APIRouter, BackgroundTasks, UploadFile, File, Form, HTTPException, Depends
from fastapi.responses import JSONResponse, StreamingResponse
from sqlalchemy.orm import Session

from app.schemas import (
    UploadResponse, ChatRequest, ChatResponse, ChatHistoryItem,
    ResumeStatus, HealthResponse, ResumeAnalyticsResponse,ResumeSummuryResponse
)
from app.pdf_parser import parse_resume
from app.vector_store import ResumeVectorStore
from app.rag_engine import RAGEngine
from app.config import get_settings, Settings
from app.database import get_db
from app.models import ChatHistory

logger = logging.getLogger(__name__)
router = APIRouter()


# ─── Dependency injection ─────────────────────────────────
# These are created once at startup and shared across requests

_vector_store: ResumeVectorStore = None
_rag_engine: RAGEngine = None


def get_vector_store() -> ResumeVectorStore:
    if _vector_store is None:
        raise HTTPException(status_code=503, detail="Vector store service not initialized. Try again shortly.")
    return _vector_store


def get_rag_engine() -> RAGEngine:
    if _rag_engine is None:
        raise HTTPException(status_code=503, detail="RAG engine service not initialized. Try again shortly.")
    return _rag_engine


def init_services(settings: Settings):
    """Called once at app startup to initialise heavy services."""
    global _vector_store, _rag_engine

    Path(settings.upload_dir).mkdir(parents=True, exist_ok=True)
    Path(settings.chroma_path).mkdir(parents=True, exist_ok=True)

    _vector_store = ResumeVectorStore(
        persist_path=settings.chroma_path,
        embedding_model=settings.embedding_model
    )

    _rag_engine = RAGEngine(
        groq_api_key=settings.groq_api_key,
        model_name=settings.model_name
    )

    logger.info("All services initialised.")


# ─── Endpoints ────────────────────────────────────────────

@router.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check(
    settings: Settings = Depends(get_settings)
):
    """
    Simple health check endpoint.
    Call this first to verify the server is running.
    """
    return HealthResponse(
        status="ok",
        version="1.0.0",
        model=settings.model_name
    )


@router.post("/upload", response_model=UploadResponse, tags=["Resume"])
async def upload_resume(
    file: UploadFile = File(..., description="PDF resume file"),
    user_id: str = Form(default="", description="User ID (auto-generated if empty)"),
    settings: Settings = Depends(get_settings),
    store: ResumeVectorStore = Depends(get_vector_store)
):
    """
    Upload a PDF resume and process it for Q&A.
    """
    # Validate file type
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

    # Validate size
    content = await file.read()
    size_mb = len(content) / (1024 * 1024)
    if size_mb > settings.max_upload_size_mb:
        raise HTTPException(
            status_code=413,
            detail=f"File too large ({size_mb:.1f} MB). Max is {settings.max_upload_size_mb} MB."
        )

    # Generate user_id if not provided
    if not user_id.strip():
        user_id = str(uuid.uuid4())[:8]

    # Save PDF to disk
    save_path = Path(settings.upload_dir) / f"{user_id}_{file.filename}"
    with open(save_path, "wb") as f:
        f.write(content)
    logger.info(f"Saved PDF: {save_path}")

    # Parse PDF → chunks
    try:
        chunks = parse_resume(
            str(save_path),
            chunk_size=settings.chunk_size,
            overlap=settings.chunk_overlap
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error(f"PDF parsing failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to parse PDF.")

    # Embed + store
    chunks_stored = store.add_resume(user_id=user_id, chunks=chunks)

    return UploadResponse(
        user_id=user_id,
        filename=file.filename,
        chunks_stored=chunks_stored,
        message=f"Resume uploaded successfully. Use user_id='{user_id}' to chat."
    )


@router.post("/chat", response_model=ChatResponse, tags=["Chat"])
async def chat(
    request: ChatRequest,
    store: ResumeVectorStore = Depends(get_vector_store),
    engine: RAGEngine = Depends(get_rag_engine),
    db: Session = Depends(get_db)
):
    """
    Ask a question about an uploaded resume.
    """
    # Check if resume exists
    chunk_count = store.get_chunk_count(request.user_id)
    if chunk_count == 0:
        raise HTTPException(
            status_code=404,
            detail=f"No resume found for user_id='{request.user_id}'. Upload a resume first."
        )

    # Retrieve relevant chunks
    relevant_chunks = store.search(
        user_id=request.user_id,
        query=request.question,
        k=4
    )

    # Convert chat history to dict format
    history_dicts = [
        {
            "role": msg.role,
            "content": msg.content
        }
    
        for msg in (request.chat_history or [])
    ]

    # Generate answer using RAG
    try:
        answer = engine.answer(
            query=request.question,
            context_chunks=relevant_chunks,
            chat_history=history_dicts
        )
    except Exception as e:
        logger.error(f"LLM call failed: {e}")
        raise HTTPException(
            status_code=502,
            detail="AI model call failed."
        )

    # Save chat history
    try:
        chat_entry = ChatHistory(
            user_id=request.user_id,
            question=request.question,
            answer=answer
        )
        db.add(chat_entry)
        db.commit()
        logger.info(f"Saved chat history for user '{request.user_id}'")
    except Exception as e:
        logger.error(f"Failed to save chat history: {e}")

    # Return response
    return ChatResponse(
        user_id=request.user_id,
        question=request.question,
        answer=answer,
        chunks_used=len(relevant_chunks),
        timestamp=datetime.now(timezone.utc)
    )


# =========================Resume summury and analytics endpoint=====================

@router.get(
    "/resume-summury/{user_id}",
    response_model=ResumeSummuryResponse,
    tags=["Resume"]
)
async def resume_summury(
    user_id: str,
    store: ResumeVectorStore = Depends(get_vector_store),
    engine: RAGEngine = Depends(get_rag_engine)
):
    """
    Generate resume summary.
    """
    chunk_count = store.get_chunk_count(user_id)

    if chunk_count == 0:
        raise HTTPException(
            status_code=404,
            detail="Resume Not Found."
        )

    chunks = store.search(
        user_id=user_id,
        query="Give complete resume overview including skills projects education experience",
        k=10
    )

    try:
        summary_text = engine.answer(
            query="Provide a concise professional summary of this resume covering skills, experience, education and key projects.",
            context_chunks=chunks,
            chat_history=[]
        )
    except Exception as e:
        logger.error(f"Summary generation failed: {e}")
        raise HTTPException(status_code=502, detail="Failed to generate resume summary.")

    return ResumeSummuryResponse(
        user_id=user_id,
        summary=summary_text,
        key_skills=[],
        total_chunks_analyzed=len(chunks)
    )




@router.post("/chat/stream", tags=["Chat"])
async def chat_stream(
    request: ChatRequest,
    background_tasks: BackgroundTasks,
    store: ResumeVectorStore = Depends(get_vector_store),
    engine: RAGEngine = Depends(get_rag_engine),
    db: Session = Depends(get_db)
):
    """
    Stream AI responses token-by-token.
    """
    # Check resume exists
    chunk_count = store.get_chunk_count(request.user_id)
    if chunk_count == 0:
        raise HTTPException(
            status_code=404,
            detail="No resume found. Upload resume first."
        )

    # Retrieve chunks
    relevant_chunks = store.search(
        user_id=request.user_id,
        query=request.question,
        k=4
    )

    # Convert history
    history_dicts = [
        {
            "role": msg.role,
            "content": msg.content
        }
        for msg in (request.chat_history or [])
    ]

    # Collect tokens to persist full answer after streaming
    collected_tokens: list[str] = []

    # Streaming generator
    def generate():
        try:
            for token in engine.stream_answer(
                query=request.question,
                context_chunks=relevant_chunks,
                chat_history=history_dicts
            ):
                collected_tokens.append(token)
                yield token

        except Exception as e:
            logger.exception("Streaming endpoint failed")
            yield f"\n[ERROR] {str(e)}"

    # Background task: save full streamed answer to chat history
    def save_stream_history():
        try:
            full_answer = "".join(collected_tokens)
            chat_entry = ChatHistory(
                user_id=request.user_id,
                question=request.question,
                answer=full_answer
            )
            db.add(chat_entry)
            db.commit()
            logger.info(f"Saved streamed chat history for user '{request.user_id}'")
        except Exception as e:
            logger.error(f"Failed to save streamed chat history: {e}")

    background_tasks.add_task(save_stream_history)

    return StreamingResponse(
        generate(),
        media_type="text/event-stream"
    )


@router.get("/chat-history/{user_id}", response_model=List[ChatHistoryItem], tags=["Chat"])
async def get_chat_history(
    user_id: str,
    db: Session = Depends(get_db)
):
    """
    Getting saved history of a user
    """
    chats = (
        db.query(ChatHistory)
        .filter(ChatHistory.user_id == user_id)
        .order_by(ChatHistory.created_at.asc())
        .all()
    )

    return [
        ChatHistoryItem(
            question=chat.question,
            answer=chat.answer,
            created_at=chat.created_at
        )
        for chat in chats
    ]


@router.get("/status/{user_id}", response_model=ResumeStatus, tags=["Resume"])
async def get_status(
    user_id: str,
    store: ResumeVectorStore = Depends(get_vector_store)
):
    """Check if a resume is uploaded for the given user_id."""
    count = store.get_chunk_count(user_id)
    return ResumeStatus(
        user_id=user_id,
        has_resume=count > 0,
        chunks_stored=count,
        message="Resume found." if count > 0 else "No resume uploaded yet."
    )


@router.delete("/resume/{user_id}", tags=["Resume"])
async def delete_resume(
    user_id: str,
    store: ResumeVectorStore = Depends(get_vector_store),
    settings: Settings = Depends(get_settings)
):
    """Delete all stored data for a user (vector chunks + uploaded file)."""
    store.delete_resume(user_id)

    # Also delete saved PDFs for this user
    upload_dir = Path(settings.upload_dir)
    deleted_files = []
    for f in upload_dir.glob(f"{user_id}_*.pdf"):
        f.unlink()
        deleted_files.append(f.name)

    return {
        "message": f"Deleted resume data for user_id='{user_id}'",
        "files_deleted": deleted_files
    }