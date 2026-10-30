"""Tests for direct-LLM versus RAG latency comparison."""
from types import SimpleNamespace

import pytest

from rag.benchmark import compare_llm_only_vs_rag


def test_comparison_reports_both_answers_and_latency_fields():
    class FakeRetriever:
        def retrieve_evidence(self, query, top_k=5):
            return [SimpleNamespace(score=0.8)], "Medium"

    class FakeLLM:
        def generate_text_response(self, **kwargs):
            assert "general model knowledge" in kwargs["system_prompt"]
            return "Direct model answer"

    class FakeAgent:
        def run_workflow(self, query, top_k=5):
            return SimpleNamespace(
                response="Grounded RAG answer",
                structured_answer=None,
                sources=[{"document_name": "case.txt"}],
                status="Completed",
            )

    result = compare_llm_only_vs_rag(
        "Summarize the evidence", FakeRetriever(), FakeLLM(), FakeAgent(), top_k=3
    )

    assert result["llm_only"]["answer"] == "Direct model answer"
    assert result["rag"]["answer"] == "Grounded RAG answer"
    assert result["rag"]["retrieved_count"] == 1
    assert result["rag"]["source_count"] == 1
    assert result["llm_only"]["latency_seconds"] >= 0
    assert result["rag"]["latency_seconds"] >= 0
    assert result["rag"]["retrieval_latency_seconds"] >= 0


def test_comparison_rejects_empty_query():
    with pytest.raises(ValueError, match="non-empty query"):
        compare_llm_only_vs_rag(" ", object(), object(), object())
