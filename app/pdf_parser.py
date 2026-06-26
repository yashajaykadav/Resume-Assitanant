"""
app/pdf_parser.py - ENHANCED VERSION
Better PDF handling, more section detection, and smarter chunking.
"""
import fitz
import re
from pathlib import Path
from typing import List, Dict, Optional
import logging
import hashlib
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)

# Enhanced section headers with variations
SECTION_HEADERS = {
    "skills": ["skills", "technical skills", "core competencies", "technologies"],
    "experience": ["experience", "work experience", "employment history", "work history"],
    "education": ["education", "academic background", "qualifications"],
    "projects": ["projects", "personal projects", "key projects"],
    "certifications": ["certifications", "certificates", "professional certifications"],
    "achievements": ["achievements", "awards", "honors"],
    "summary": ["summary", "professional summary", "profile", "about me"],
    "languages": ["languages", "language skills"],
    "publications": ["publications", "papers"],
}

def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extract text from all PDF pages with better error handling.
    """
    try:
        doc = fitz.open(pdf_path)
        
        # Check if PDF is encrypted
        if doc.is_encrypted:
            raise ValueError("PDF is encrypted/password protected")
        
        pages_text = []
        
        for page_num, page in enumerate(doc):
            text = page.get_text("text")
            
            # If no text found, try extracting as blocks
            if not text.strip():
                blocks = page.get_text("blocks")
                text = "\n".join([block[4] for block in blocks if block[4].strip()])
            
            if text.strip():
                pages_text.append(f"--- Page {page_num + 1} ---\n{text}")
            else:
                logger.warning(f"No text found on page {page_num + 1}")
        
        doc.close()
        
        if not pages_text:
            raise ValueError("No text could be extracted from PDF")
        
        raw_text = "\n\n".join(pages_text)
        return clean_text(raw_text)
        
    except fitz.fitz.FileDataError as e:
        logger.error(f"Corrupted PDF: {e}")
        raise ValueError("PDF file appears to be corrupted")
    except Exception as e:
        logger.error(f"PDF extraction failed: {e}")
        raise

def clean_text(text: str) -> str:
    """
    Enhanced text cleaning with better artifact removal.
    """
    # Remove null bytes and control characters
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)
    
    # Fix hyphenated line breaks
    text = re.sub(r'(\w+)-\n(\w+)', r'\1\2', text)
    
    # Normalize spaces and line breaks
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n\s*\n', '\n\n', text)
    
    # Remove page numbers and headers/footers (common patterns)
    text = re.sub(r'\n\s*\d+\s*\n', '\n', text)
    text = re.sub(r'Page \d+ of \d+', '', text, flags=re.IGNORECASE)
    
    # Remove email addresses (optional - for privacy)
    # text = re.sub(r'\S+@\S+\.\S+', '[EMAIL]', text)
    
    # Remove phone numbers (optional - for privacy)
    # text = re.sub(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', '[PHONE]', text)
    
    return text.strip()

def split_into_sections(text: str) -> List[str]:
    """
    Enhanced section splitting with fuzzy matching and nested sections.
    """
    lines = text.splitlines()
    sections = {}
    current_section = "header"
    sections[current_section] = []
    
    # First pass: identify all section boundaries
    for line in lines:
        stripped = line.strip().lower()
        
        # Check if line matches any section header
        matched_section = None
        for section, headers in SECTION_HEADERS.items():
            if any(stripped == header or stripped.startswith(header) for header in headers):
                matched_section = section
                break
        
        if matched_section:
            current_section = matched_section
            if current_section not in sections:
                sections[current_section] = []
            sections[current_section].append(line)
        else:
            if current_section in sections:
                sections[current_section].append(line)
            else:
                sections[current_section] = [line]
    
    # Convert to list format with section headers preserved
    section_texts = []
    for section_name, content_lines in sections.items():
        if content_lines:
            section_text = "\n".join(content_lines).strip()
            if section_text:
                section_texts.append(section_text)
    
    return section_texts

def chunk_text(
    text: str,
    chunk_size: int = 500,
    overlap: int = 50,
    min_chunk_size: int = 50
) -> List[str]:
    """
    Create overlapping chunks with smart boundaries.
    Uses sentence-aware chunking for better semantic coherence.
    """
    # Split into sentences
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    chunks = []
    current_chunk = []
    current_length = 0
    
    for sentence in sentences:
        sentence_length = len(sentence.split())
        
        # If adding this sentence exceeds chunk size, save current chunk
        if current_length + sentence_length > chunk_size and current_chunk:
            chunk_text = ' '.join(current_chunk).strip()
            if len(chunk_text.split()) >= min_chunk_size:
                chunks.append(chunk_text)
            
            # Keep overlap sentences
            overlap_words = chunk_size - overlap
            overlap_sentences = []
            overlap_length = 0
            
            for s in reversed(current_chunk):
                s_len = len(s.split())
                if overlap_length + s_len <= overlap_words:
                    overlap_sentences.insert(0, s)
                    overlap_length += s_len
                else:
                    break
            
            current_chunk = overlap_sentences
            current_length = overlap_length
        
        current_chunk.append(sentence)
        current_length += sentence_length
    
    # Add final chunk
    if current_chunk:
        chunk_text = ' '.join(current_chunk).strip()
        if len(chunk_text.split()) >= min_chunk_size:
            chunks.append(chunk_text)
    
    return chunks if chunks else [text[:500]]  # Fallback

def extract_metadata(pdf_path: str) -> Dict[str, str]:
    """
    Extract PDF metadata for additional context.
    """
    try:
        doc = fitz.open(pdf_path)
        metadata = doc.metadata
        doc.close()
        
        return {
            "title": metadata.get("title", ""),
            "author": metadata.get("author", ""),
            "subject": metadata.get("subject", ""),
            "keywords": metadata.get("keywords", ""),
        }
    except Exception as e:
        logger.warning(f"Could not extract metadata: {e}")
        return {}

def parse_resume(
    pdf_path: str,
    chunk_size: int = 500,
    overlap: int = 50
) -> List[str]:
    """
    Full pipeline with enhanced features.
    """
    pdf_file = Path(pdf_path)
    
    if not pdf_file.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    
    # Check file size
    file_size_mb = pdf_file.stat().st_size / (1024 * 1024)
    if file_size_mb > 10:
        logger.warning(f"Large PDF file: {file_size_mb:.2f} MB")
    
    # Extract text
    try:
        raw_text = extract_text_from_pdf(pdf_path)
    except Exception as e:
        logger.error(f"Failed to extract text: {e}")
        raise ValueError(f"Could not extract text from PDF: {e}")
    
    if not raw_text.strip():
        raise ValueError("PDF contains no extractable text (might be scanned/image-based)")
    
    # Extract metadata (optional)
    metadata = extract_metadata(pdf_path)
    if metadata.get("title"):
        logger.info(f"Processing resume: {metadata['title']}")
    
    # Split into semantic sections
    sections = split_into_sections(raw_text)
    
    if not sections:
        # Fallback: chunk entire text
        logger.warning("No sections detected, chunking entire document")
        sections = [raw_text]
    
    all_chunks = []
    
    # Chunk each section separately with section headers preserved
    for section in sections:
        section_chunks = chunk_text(
            section,
            chunk_size=chunk_size,
            overlap=overlap
        )
        all_chunks.extend(section_chunks)
    
    # Remove duplicate chunks (exact duplicates)
    seen = set()
    unique_chunks = []
    for chunk in all_chunks:
        chunk_hash = hashlib.md5(chunk.encode()).hexdigest()
        if chunk_hash not in seen:
            seen.add(chunk_hash)
            unique_chunks.append(chunk)
    
    logger.info(f"Generated {len(unique_chunks)} unique chunks from {len(sections)} sections")
    
    return unique_chunks

# Add missing import
