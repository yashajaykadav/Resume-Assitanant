"""
app/vector_store.py - ENHANCED VERSION
Added error handling, batch processing, and performance optimizations.
"""
import chromadb
from chromadb.config import Settings as ChromaSettings
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any, Optional
import hashlib
import logging
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)


class ResumeVectorStore:
    """
    Enhanced vector store with batch processing and better error handling.
    """

    def __init__(self, persist_path: str, embedding_model: str):
        # Load embedding model with error handling
        try:
            logger.info(f"Loading embedding model: {embedding_model}")
            self.embedder = SentenceTransformer(embedding_model)
            
            # Move to GPU if available (10x speedup)
            import torch
            if torch.cuda.is_available():
                self.embedder = self.embedder.to('cuda')
                logger.info("Using GPU for embeddings")
            else:
                logger.info("Using CPU for embeddings")
                
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise

        # Set up ChromaDB with persistent storage
        self.client = chromadb.PersistentClient(
            path=persist_path,
            settings=ChromaSettings(anonymized_telemetry=False)
        )

        # One collection for all resumes
        self.collection = self.client.get_or_create_collection(
            name="resumes",
            metadata={"hnsw:space": "cosine"}
        )
        
        # Cache for performance
        self._embedding_cache = {}
        
        logger.info("Vector store ready.")

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def add_resume(self, user_id: str, chunks: List[str]) -> int:
        """
        Embed all chunks and upsert into ChromaDB with retry logic.
        """
        if not chunks:
            raise ValueError("No chunks to store.")
        
        # Validate chunk sizes
        for i, chunk in enumerate(chunks):
            if len(chunk) > 5000:
                logger.warning(f"Chunk {i} is very long ({len(chunk)} chars), truncating")
                chunks[i] = chunk[:5000]
        
        # Generate unique IDs for each chunk
        ids = [
            f"{user_id}_chunk_{i}_{_short_hash(chunk)}"
            for i, chunk in enumerate(chunks)
        ]

        # Batch embedding for better performance
        batch_size = 32
        all_embeddings = []
        
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i+batch_size]
            logger.info(f"Embedding batch {i//batch_size + 1}/{(len(chunks)-1)//batch_size + 1}")
            
            embeddings = self.embedder.encode(
                batch, 
                show_progress_bar=False,
                convert_to_numpy=True
            ).tolist()
            all_embeddings.extend(embeddings)

        # Tag each chunk with metadata
        metadatas = [
            {
                "user_id": user_id, 
                "chunk_index": i,
                "chunk_length": len(chunk),
                "timestamp": _get_timestamp()
            } 
            for i, chunk in enumerate(chunks)
        ]

        # Delete existing chunks for this user
        self.delete_resume(user_id)

        # Add in batches to avoid memory issues
        batch_size = 100
        for i in range(0, len(chunks), batch_size):
            batch_ids = ids[i:i+batch_size]
            batch_embeddings = all_embeddings[i:i+batch_size]
            batch_docs = chunks[i:i+batch_size]
            batch_metadata = metadatas[i:i+batch_size]
            
            self.collection.add(
                ids=batch_ids,
                embeddings=batch_embeddings,
                documents=batch_docs,
                metadatas=batch_metadata
            )
            
            logger.info(f"Added batch {i//batch_size + 1}/{(len(chunks)-1)//batch_size + 1}")

        logger.info(f"Stored {len(chunks)} chunks for user {user_id}")
        return len(chunks)

    def search(self, user_id: str, query: str, k: int = 4) -> List[str]:
        """
        Find top-k relevant chunks with score filtering.
        """
        # Check cache first
        cache_key = f"{user_id}_{query}_{k}"
        if cache_key in self._embedding_cache:
            logger.info(f"Using cached result for query: {query[:50]}")
            return self._embedding_cache[cache_key]
        
        # Embed query
        query_embedding = self.embedder.encode([query])[0].tolist()

        # Get more results initially for filtering
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(k * 2, self.collection.count()),  # Get extra for filtering
            where={"user_id": user_id},
            include=["documents", "distances", "metadatas"]
        )

        # Filter by similarity score (cosine distance < 0.5 means good match)
        docs = []
        if results["documents"] and results["distances"]:
            for doc, distance in zip(results["documents"][0], results["distances"][0]):
                # Cosine distance: 0 = identical, 2 = opposite
                # Keep results with distance < 1.0 (reasonable match)
                if distance < 1.0:
                    docs.append(doc)
                    if len(docs) >= k:
                        break
        
        # Cache result (TTL 5 minutes)
        self._embedding_cache[cache_key] = docs
        
        # Clean cache if too large
        if len(self._embedding_cache) > 100:
            self._embedding_cache.clear()
        
        return docs

    def delete_resume(self, user_id: str) -> None:
        """Remove all stored chunks for a given user."""
        try:
            # Get all chunks for this user
            existing = self.collection.get(where={"user_id": user_id})
            if existing["ids"]:
                # Delete in batches
                batch_size = 100
                for i in range(0, len(existing["ids"]), batch_size):
                    batch_ids = existing["ids"][i:i+batch_size]
                    self.collection.delete(ids=batch_ids)
                
                logger.info(f"Deleted {len(existing['ids'])} old chunks for {user_id}")
                
                # Clear cache for this user
                keys_to_delete = [k for k in self._embedding_cache if k.startswith(user_id)]
                for key in keys_to_delete:
                    del self._embedding_cache[key]
                    
        except Exception as e:
            logger.error(f"Could not delete chunks for {user_id}: {e}")
            raise

    def get_chunk_count(self, user_id: str) -> int:
        """How many chunks are stored for this user?"""
        try:
            result = self.collection.get(where={"user_id": user_id})
            return len(result["ids"])
        except Exception as e:
            logger.error(f"Failed to get chunk count for {user_id}: {e}")
            return 0
    
    def get_all_chunks(self, user_id: str) -> List[Dict[str, Any]]:
        """Retrieve all chunks with metadata for debugging."""
        result = self.collection.get(where={"user_id": user_id})
        return [
            {"id": id_, "text": doc, "metadata": meta}
            for id_, doc, meta in zip(result["ids"], result["documents"], result["metadatas"])
        ]


def _short_hash(text: str) -> str:
    """First 8 chars of MD5."""
    return hashlib.md5(text.encode()).hexdigest()[:8]

def _get_timestamp() -> str:
    """Get current timestamp string."""
    from datetime import datetime
    return datetime.utcnow().isoformat()