from typing import List, Dict, Any, Tuple
from models.schemas import DocumentChunk
from rag.vector_store import VectorStoreManager
from rag.embeddings import EmbeddingService

class RAGRetriever:
    """Retriever class responsible for executing vector similarity searches

    and scoring retrieved evidence for the Legal Research Agent.
    """

    def __init__(self, vector_store: VectorStoreManager, embedding_service: EmbeddingService):
        self.vector_store = vector_store
        self.embedding_service = embedding_service

    def retrieve_evidence(self, query: str, top_k: int = 5) -> Tuple[List[DocumentChunk], str]:
        chunks = self.vector_store.search(query, top_k=top_k, embedding_service=self.embedding_service)

        if not chunks:
            return [], "Insufficient"

        # Calculate average similarity score to evaluate evidence strength
        avg_score = sum(c.score for c in chunks) / len(chunks)
        max_score = max(c.score for c in chunks)

        if max_score > 0.65 or avg_score > 0.5:
            strength = "High"
        elif max_score > 0.4 or avg_score > 0.3:
            strength = "Medium"
        elif max_score > 0.2:
            strength = "Low"
        else:
            strength = "Insufficient"

        return chunks, strength

    def format_chunks_for_context(self, chunks: List[DocumentChunk]) -> str:
        if not chunks:
            return "NO RELEVANT DOCUMENTS FOUND."

        formatted_blocks = []
        for idx, chunk in enumerate(chunks, 1):
            block = (
                f"--- EVIDENCE ITEM [{idx}] ---\n"
                f"Source File: {chunk.document_name}\n"
                f"Document Type: {chunk.document_type}\n"
                f"Page Number: {chunk.page_number}\n"
                f"Chunk ID: {chunk.chunk_id}\n"
                f"Relevance Score: {chunk.score}\n"
                f"Content:\n{chunk.text}\n"
            )
            formatted_blocks.append(block)

        return "\n\n".join(formatted_blocks)
