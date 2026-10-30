"""Configuration and backend-selection tests for vector storage."""
import pytest

from rag.vector_store import VectorStoreManager


def test_vector_store_rejects_unknown_backend(tmp_path):
    with pytest.raises(ValueError, match="VECTOR_STORE_BACKEND"):
        VectorStoreManager(persistent_dir=tmp_path, backend="unknown")


def test_chroma_remains_the_default_backend(tmp_path):
    manager = VectorStoreManager(persistent_dir=tmp_path, backend="chroma")
    assert manager.backend == "chroma"
    assert manager.get_stats()["total_chunks"] == 0


def test_faiss_backend_reports_missing_optional_dependency(tmp_path, monkeypatch):
    import builtins

    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name == "faiss":
            raise ImportError("faiss intentionally unavailable in this test")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    with pytest.raises(RuntimeError, match="faiss-cpu"):
        VectorStoreManager(persistent_dir=tmp_path, backend="faiss")
