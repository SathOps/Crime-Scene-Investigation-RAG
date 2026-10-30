import hashlib
import json
from pathlib import Path
from typing import List, Dict, Any

import chromadb
from chromadb.config import Settings
import numpy as np

from models.schemas import DocumentChunk
from rag.embeddings import EmbeddingService
from config import VECTORSTORE_DIR, DEFAULT_VECTOR_STORE_BACKEND


class VectorStoreManager:
    """Persistent vector storage with ChromaDB (default) or optional FAISS CPU."""

    def __init__(self, persistent_dir: Path = VECTORSTORE_DIR, backend: str = None):
        self.persistent_dir = Path(persistent_dir)
        self.backend = (backend or DEFAULT_VECTOR_STORE_BACKEND).lower()
        if self.backend not in {"chroma", "faiss"}:
            raise ValueError("VECTOR_STORE_BACKEND must be either 'chroma' or 'faiss'.")
        self.collection_name = "legal_case_chunks"
        self.persistent_dir.mkdir(parents=True, exist_ok=True)
        if self.backend == "faiss":
            self._init_faiss()
        else:
            self._init_chroma()

    def _init_chroma(self):
        self.client = chromadb.PersistentClient(
            path=str(self.persistent_dir / "chroma"),
            settings=Settings(allow_reset=True, anonymized_telemetry=False),
        )
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def _init_faiss(self):
        try:
            import faiss
        except ImportError as exc:
            raise RuntimeError(
                "FAISS backend selected but faiss-cpu is not installed. "
                "Install it with: python -m pip install faiss-cpu"
            ) from exc
        self._faiss = faiss
        self._faiss_index_path = self.persistent_dir / "legal_case_chunks.faiss"
        self._faiss_records_path = self.persistent_dir / "legal_case_chunks.json"
        self._faiss_records: Dict[str, Dict[str, Any]] = {}
        if self._faiss_index_path.exists() and self._faiss_records_path.exists():
            self.index = faiss.read_index(str(self._faiss_index_path))
            self._faiss_records = json.loads(self._faiss_records_path.read_text(encoding="utf-8"))
        else:
            self.index = None

    @staticmethod
    def _faiss_id(chunk_id: str) -> int:
        raw = hashlib.blake2b(chunk_id.encode("utf-8"), digest_size=8).digest()
        return int.from_bytes(raw, byteorder="big", signed=True)

    @staticmethod
    def _normalized_matrix(vectors):
        matrix = np.asarray(vectors, dtype="float32")
        if matrix.ndim != 2:
            raise ValueError("Embedding vectors must form a two-dimensional matrix.")
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        return matrix / np.maximum(norms, 1e-12)

    def _save_faiss(self):
        if self.index is not None:
            self._faiss.write_index(self.index, str(self._faiss_index_path))
        self._faiss_records_path.write_text(
            json.dumps(self._faiss_records, ensure_ascii=False),
            encoding="utf-8",
        )

    def add_chunks(self, chunks: List[DocumentChunk], embedding_service: EmbeddingService):
        if not chunks:
            return
        texts = [chunk.text for chunk in chunks]
        embeddings = embedding_service.embed_texts(texts)
        if len(embeddings) != len(chunks):
            raise ValueError("Embedding service returned a different number of vectors than chunks.")

        if self.backend == "faiss":
            matrix = self._normalized_matrix(embeddings)
            dimension = matrix.shape[1]
            if self.index is None:
                self.index = self._faiss.IndexIDMap2(self._faiss.IndexFlatIP(dimension))
            elif self.index.d != dimension:
                raise ValueError(
                    f"Embedding dimension changed from {self.index.d} to {dimension}; "
                    "use a separate vector-store directory for this embedding model."
                )
            for chunk, vector in zip(chunks, matrix):
                faiss_id = self._faiss_id(chunk.chunk_id)
                self.index.remove_ids(np.asarray([faiss_id], dtype="int64"))
                self._faiss_records[str(faiss_id)] = {
                    "chunk_id": chunk.chunk_id,
                    "document_name": chunk.document_name,
                    "document_type": chunk.document_type,
                    "page_number": int(chunk.page_number),
                    "case_name": chunk.case_name,
                    "text": chunk.text,
                }
                self.index.add_with_ids(
                    np.asarray([vector], dtype="float32"),
                    np.asarray([faiss_id], dtype="int64"),
                )
            self._save_faiss()
            return

        ids = [chunk.chunk_id for chunk in chunks]
        metadatas = [
            {
                "document_name": chunk.document_name,
                "document_type": chunk.document_type,
                "page_number": int(chunk.page_number),
                "case_name": chunk.case_name,
            }
            for chunk in chunks
        ]
        batch_size = 100
        for start in range(0, len(chunks), batch_size):
            end = min(start + batch_size, len(chunks))
            self.collection.upsert(
                ids=ids[start:end],
                embeddings=embeddings[start:end],
                documents=texts[start:end],
                metadatas=metadatas[start:end],
            )

    def search(self, query: str, top_k: int, embedding_service: EmbeddingService) -> List[DocumentChunk]:
        if top_k <= 0:
            return []
        query_embedding = embedding_service.embed_query(query)
        if not query_embedding:
            return []

        if self.backend == "faiss":
            if self.index is None or self.index.ntotal == 0:
                return []
            vector = self._normalized_matrix([query_embedding])
            if vector.shape[1] != self.index.d:
                raise ValueError("Query embedding dimension does not match the indexed documents.")
            scores, ids = self.index.search(vector, min(top_k, self.index.ntotal))
            retrieved = []
            for score, faiss_id in zip(scores[0], ids[0]):
                if int(faiss_id) == -1:
                    continue
                record = self._faiss_records.get(str(int(faiss_id)))
                if not record:
                    continue
                retrieved.append(DocumentChunk(
                    chunk_id=record["chunk_id"],
                    document_name=record["document_name"],
                    document_type=record["document_type"],
                    page_number=int(record["page_number"]),
                    case_name=record.get("case_name", "Unknown Case"),
                    text=record["text"],
                    score=round(max(0.0, min(1.0, float(score))), 3),
                ))
            return retrieved

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, self.collection.count()),
            include=["documents", "metadatas", "distances"],
        )
        retrieved: List[DocumentChunk] = []
        if results and results.get("documents") and results["documents"][0]:
            for idx, text in enumerate(results["documents"][0]):
                distance = results["distances"][0][idx]
                metadata = results["metadatas"][0][idx]
                retrieved.append(DocumentChunk(
                    chunk_id=results["ids"][0][idx],
                    document_name=metadata.get("document_name", "Unknown"),
                    document_type=metadata.get("document_type", "Legal Document"),
                    page_number=int(metadata.get("page_number", 1)),
                    case_name=metadata.get("case_name", "Unknown Case"),
                    text=text,
                    score=round(max(0.0, 1.0 - distance), 3),
                ))
        return retrieved

    def get_stats(self) -> Dict[str, Any]:
        if self.backend == "faiss":
            records = list(self._faiss_records.values())
            documents = sorted({record["document_name"] for record in records})
            return {
                "total_chunks": len(records),
                "total_documents": len(documents),
                "documents_list": documents,
            }
        count = self.collection.count()
        results = self.collection.get(include=["metadatas"])
        unique_docs = {
            metadata["document_name"]
            for metadata in (results.get("metadatas") or [])
            if metadata and "document_name" in metadata
        }
        return {
            "total_chunks": count,
            "total_documents": len(unique_docs),
            "documents_list": sorted(unique_docs),
        }

    def clear_store(self):
        if self.backend == "faiss":
            self.index = None
            self._faiss_records = {}
            for path in (self._faiss_index_path, self._faiss_records_path):
                if path.exists():
                    path.unlink()
            return
        try:
            self.client.delete_collection(self.collection_name)
        except Exception:
            pass
        self._init_chroma()
