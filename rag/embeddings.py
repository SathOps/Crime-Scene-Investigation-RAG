import os
from typing import List
import numpy as np

class EmbeddingService:
    """Modular embedding generator supporting local sentence-transformers, Google Gemini, and OpenAI."""

    def __init__(self, provider: str = "local", api_key: str = ""):
        self.provider = provider.lower()
        self.api_key = api_key
        self._local_model = None

    def _get_local_model(self):
        if self._local_model is None:
            from sentence_transformers import SentenceTransformer
            # Small, high-quality, fast local model for RAG
            self._local_model = SentenceTransformer("all-MiniLM-L6-v2")
        return self._local_model

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []

        if self.provider == "gemini" and self.api_key:
            try:
                from google import genai
                client = genai.Client(api_key=self.api_key)
                response = client.models.embed_content(
                    model="text-embedding-004",
                    contents=texts
                )
                return [e.values for e in response.embeddings]
            except Exception as e:
                print(f"[Warning] Gemini embedding failed: {e}. Falling back to local SentenceTransformer.")

        elif self.provider == "openai" and self.api_key:
            try:
                from openai import OpenAI
                client = OpenAI(api_key=self.api_key)
                res = client.embeddings.create(
                    input=texts,
                    model="text-embedding-3-small"
                )
                return [item.embedding for item in res.data]
            except Exception as e:
                print(f"[Warning] OpenAI embedding failed: {e}. Falling back to local SentenceTransformer.")

        # Default / Local Fallback
        model = self._get_local_model()
        embeddings = model.encode(texts, show_progress_bar=False)
        return embeddings.tolist()

    def embed_query(self, query: str) -> List[float]:
        res = self.embed_texts([query])
        return res[0] if res else []
