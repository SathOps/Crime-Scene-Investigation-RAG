"""Tests for labelled retrieval metrics and benchmark execution."""
from models.schemas import DocumentChunk
from rag.evaluation import benchmark_retrieval, evaluate_retrieval


def chunk(chunk_id, document_name, score=0.0):
    return DocumentChunk(
        chunk_id=chunk_id,
        document_name=document_name,
        document_type="Legal Document",
        page_number=1,
        text="Evidence excerpt",
        score=score,
    )


def test_evaluate_retrieval_calculates_ranking_metrics():
    retrieved = [
        chunk("c1", "unrelated.txt", 0.9),
        chunk("c2", "Witness_Statements.txt", 0.8),
        chunk("c3", "CCTV_Report.txt", 0.7),
    ]

    metrics = evaluate_retrieval(
        retrieved,
        relevant_document_names={"witness_statements.TXT", "cctv_report.txt"},
        k=3,
    )

    assert metrics["precision_at_k"] == 0.6667
    assert metrics["recall_at_k"] == 1.0
    assert metrics["f1_at_k"] == 0.8
    assert metrics["hit_rate"] == 1.0
    assert metrics["reciprocal_rank"] == 0.5
    assert metrics["relevant_retrieved"] == 2


def test_evaluate_retrieval_supports_chunk_id_labels_and_empty_results():
    metrics = evaluate_retrieval(
        [chunk("target-chunk", "unknown.txt")],
        relevant_chunk_ids={"target-chunk"},
        k=5,
    )
    assert metrics["precision_at_k"] == 1.0
    assert metrics["recall_at_k"] == 1.0

    empty = evaluate_retrieval([], relevant_document_names={"missing.txt"})
    assert empty["precision_at_k"] == 0.0
    assert empty["recall_at_k"] == 0.0
    assert empty["hit_rate"] == 0.0


def test_benchmark_retrieval_reports_per_query_and_aggregate_metrics():
    class FakeRetriever:
        def retrieve_evidence(self, query, top_k=5):
            if "witness" in query.lower():
                return [chunk("w1", "Witness_Statements.txt", 0.8)], "High"
            return [chunk("c1", "CCTV_Report.txt", 0.7)], "Medium"

    report = benchmark_retrieval(
        FakeRetriever(),
        [
            {"query": "Find witness account", "relevant_documents": ["witness_statements.txt"]},
            {"query": "Find CCTV record", "relevant_documents": ["cctv_report.txt"]},
        ],
        top_k=3,
    )

    assert report["query_count"] == 2
    assert report["aggregate"]["precision_at_k"] == 1.0
    assert report["aggregate"]["recall_at_k"] == 1.0
    assert len(report["results"]) == 2
    assert all("retrieval_time_seconds" in item for item in report["results"])


def test_benchmark_retrieval_rejects_blank_queries():
    import pytest

    with pytest.raises(ValueError, match="non-empty query"):
        benchmark_retrieval(object(), [{"query": " "}])


def test_default_legal_evaluation_cases_include_relevant_and_out_of_scope_queries():
    from rag.evaluation import DEFAULT_LEGAL_EVALUATION_CASES

    assert len(DEFAULT_LEGAL_EVALUATION_CASES) >= 2
    assert any(case.get("relevant_documents") for case in DEFAULT_LEGAL_EVALUATION_CASES)
    assert any(not case.get("relevant_documents") for case in DEFAULT_LEGAL_EVALUATION_CASES)
