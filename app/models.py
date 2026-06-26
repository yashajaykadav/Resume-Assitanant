"""
app/models.py - ENHANCED VERSION
"""
from sqlalchemy import Column, Integer, String, Text, DateTime, Float, Index, ForeignKey, Boolean
from datetime import datetime
from app.database import Base

class ChatHistory(Base):
    """Enhanced chat history with metrics."""
    
    __tablename__ = "chat_history"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String, index=True, nullable=False)
    question = Column(Text, nullable=False)
    answer = Column(Text, nullable=False)
    
    # Add these fields for better analytics
    chunks_used = Column(Integer, default=0)
    response_time_ms = Column(Integer, default=0)
    model_used = Column(String, default="llama-3.1-8b-instant")
    
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Add indexes for performance
    __table_args__ = (
        Index('idx_user_created', 'user_id', 'created_at'),
    )

class ResumeMetadata(Base):
    """Store resume metadata separately."""
    
    __tablename__ = "resume_metadata"
    
    id = Column(Integer, primary_key=True)
    user_id = Column(String, unique=True, index=True, nullable=False)
    filename = Column(String, nullable=False)
    file_size_bytes = Column(Integer)
    num_chunks = Column(Integer, default=0)
    upload_date = Column(DateTime, default=datetime.utcnow)
    last_accessed = Column(DateTime, onupdate=datetime.utcnow)
    
    # Resume insights
    extracted_skills = Column(Text)  # JSON string
    extracted_experience_years = Column(Float)
    ats_score = Column(Integer)  # Optional ATS score

class UserFeedback(Base):
    """Store user feedback on answers."""
    
    __tablename__ = "user_feedback"
    
    id = Column(Integer, primary_key=True)
    chat_id = Column(Integer, ForeignKey("chat_history.id"))
    user_id = Column(String, index=True)
    helpful = Column(Boolean)  # Thumbs up/down
    feedback_text = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)