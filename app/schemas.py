"""
app/schemas.py - CORRECTED & ENHANCED VERSION
Pydantic models for request/response validation.
"""
from pydantic import BaseModel, Field, validator
from typing import List, Optional
from datetime import datetime


# ─── Upload ───────────────────────────────────────────────

class UploadResponse(BaseModel):
    user_id: str
    filename: str
    chunks_stored: int
    message: str


# ─── Chat ─────────────────────────────────────────────────

class ChatMessage(BaseModel):
    role: str = Field(..., pattern="^(user|assistant)$")
    content: str = Field(..., min_length=1, max_length=2000)
    
    @validator('role')
    def validate_role(cls, v):
        if v not in ['user', 'assistant']:
            raise ValueError('role must be "user" or "assistant"')
        return v


class ChatRequest(BaseModel):
    user_id: str = Field(..., min_length=1, max_length=100, description="Unique user identifier")
    question: str = Field(..., min_length=1, max_length=1000, description="Question about the resume")
    chat_history: Optional[List[ChatMessage]] = Field(
        default=[],
        description="Previous messages for multi-turn conversation (max 10 messages)"
    )
    
    @validator('chat_history')
    def limit_history(cls, v):
        if v and len(v) > 10:
            raise ValueError('Chat history limited to 10 messages')
        return v


class ChatResponse(BaseModel):
    user_id: str
    question: str
    answer: str
    chunks_used: int
    timestamp: datetime


class ChatHistoryItem(BaseModel):
    question: str
    answer: str
    created_at: datetime
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


# ─── Resume Summary (Fixed) ──────────────────────────────

class ResumeSummuryResponse(BaseModel):
    user_id: str
    summary: str  # Fixed typo: 'summury' → 'summary'
    key_skills: List[str] = []  # Added missing field
    total_chunks_analyzed: int = 0  # Added missing field
    
    class Config:
        json_schema_extra = {
            "example": {
                "user_id": "abc123",
                "summary": "Experienced software engineer with 5 years in Python...",
                "key_skills": ["Python", "FastAPI", "Machine Learning"],
                "total_chunks_analyzed": 10
            }
        }


# ─── Resume Analytics (Enhanced) ─────────────────────────

class ResumeAnalyticsResponse(BaseModel):
    user_id: str
    skills: List[str]
    projects: List[str]
    education: List[str]
    experience: List[str]
    certifications: List[str]
    
    # Add optional analytics fields
    ats_score: Optional[int] = Field(None, ge=0, le=100)
    missing_keywords: Optional[List[str]] = []
    suggestions: Optional[List[str]] = []
    
    class Config:
        json_schema_extra = {
            "example": {
                "user_id": "abc123",
                "skills": ["Python", "SQL", "AWS"],
                "projects": ["E-commerce API", "Chatbot"],
                "education": ["BSc Computer Science"],
                "experience": ["Software Engineer at Tech Corp"],
                "certifications": ["AWS Certified"],
                "ats_score": 85,
                "missing_keywords": ["Docker", "Kubernetes"],
                "suggestions": ["Add more quantifiable achievements"]
            }
        }


# ─── Status ───────────────────────────────────────────────

class ResumeStatus(BaseModel):
    user_id: str
    has_resume: bool
    chunks_stored: int
    message: str


class HealthResponse(BaseModel):
    status: str
    version: str
    model: str
    timestamp: Optional[datetime] = None


# ─── Error ────────────────────────────────────────────────

class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.now)