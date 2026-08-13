import os
import shutil
from pathlib import Path
from typing import List, Dict, Any
import chromadb
from chromadb.config import Settings
from models.schemas import DocumentChunk
from rag.embeddings import EmbeddingService
from config import VECTORSTORE_DIR

class VectorStoreManager:
    """Manages persistent ChromaDB vector storage for legal document chunks."""

    def __init__(self, persistent_dir: Path = VECTORSTORE_DIR):
        self.persistent_dir = persistent_dir
        self.collection_name = "legal_case_chunks"
        self._init_db()

    def _init_db(self):
        self.persistent_dir.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(
            path=str(self.persistent_dir),
            settings=Settings(allow_reset=True, anonymized_telemetry=False)
        )
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )

    def add_chunks(self, chunks: List[DocumentChunk], embedding_service: EmbeddingService):
        if not chunks:
            return

        texts = [c.text for c in chunks]
        embeddings = embedding_service.embed_texts(texts)
        ids = [c.chunk_id for c in chunks]
        metadatas = [
            {
                "document_name": c.document_name,
                "document_type": c.document_type,
                "page_number": int(c.page_number),
                "case_name": c.case_name
            }
            for c in chunks
        ]

        # Add to ChromaDB in batches to prevent payload limits
        batch_size = 100
        for i in range(0, len(chunks), batch_size):
            end_idx = min(i + batch_size, len(chunks))
            self.collection.upsert(
                ids=ids[i:end_idx],
                embeddings=embeddings[i:end_idx],
                documents=texts[i:end_idx],
                metadatas=metadatas[i:end_idx]
            )

    def search(self, query: str, top_k: int, embedding_service: EmbeddingService) -> List[DocumentChunk]:
        query_embedding = embedding_service.embed_query(query)
        if not query_embedding:
            return []

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )

        retrieved: List[DocumentChunk] = []
        if results and results.get("documents") and results["documents"][0]:
            docs = results["documents"][0]
            ids = results["ids"][0]
            metadatas = results["metadatas"][0]
            distances = results["distances"][0]

            for idx in range(len(docs)):
                # Chroma cosine distance range: 0.0 (identical) to 2.0. Convert to similarity score.
                dist = distances[idx] if idx < len(distances) else 1.0
                score = round(max(0.0, 1.0 - dist), 3)
                meta = metadatas[idx]

                chunk = DocumentChunk(
                    chunk_id=ids[idx],
                    document_name=meta.get("document_name", "Unknown"),
                    document_type=meta.get("document_type", "Legal Document"),
                    page_number=int(meta.get("page_number", 1)),
                    case_name=meta.get("case_name", "Unknown Case"),
                    text=docs[idx],
                    score=score
                )
                retrieved.append(chunk)

        return retrieved

    def get_stats(self) -> Dict[str, Any]:
        count = self.collection.count()
        # Retrieve unique documents
        results = self.collection.get(include=["metadatas"])
        unique_docs = set()
        if results and results.get("metadatas"):
            for meta in results["metadatas"]:
                if meta and "document_name" in meta:
                    unique_docs.add(meta["document_name"])

        return {
            "total_chunks": count,
            "total_documents": len(unique_docs),
            "documents_list": sorted(list(unique_docs))
        }

    def clear_store(self):
        try:
            self.client.reset()
        except Exception:
            try:
                self.client.delete_collection(self.collection_name)
            except Exception:
                pass
        self._init_db()

