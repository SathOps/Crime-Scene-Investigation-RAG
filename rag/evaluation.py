"""Offline retrieval evaluation utilities for the legal-case RAG pipeline.

Ground-truth relevance is supplied by the evaluator (document names and/or chunk IDs).
Similarity scores are never treated as ground truth.
"""
from __future__ import annotations

import re
import time
from statistics import mean
from typing import Any, Dict, Iterable, List, Mapping, Sequence


def _value(item: Any, key: str, default: Any = None) -> Any:
    if isinstance(item, Mapping):
        return item.get(key, default)
    return getattr(item, key, default)


def _normalize_name(value: Any) -> str:
    return re.sub(r"\\s+", " ", str(value or "").strip().casefold())


def evaluate_retrieval(
    retrieved: Sequence[Any],
    relevant_document_names: Iterable[str] = (),
    relevant_chunk_ids: Iterable[str] = (),
    k: int | None = None,
    retrieval_time_seconds: float | None = None,
) -> Dict[str, Any]:
    """Calculate precision@k, recall@k, F1@k, hit rate and reciprocal rank.

    A retrieved item is relevant when its normalized document name is in the
    supplied document labels or its chunk ID is in the supplied chunk labels.
    """
    limit = len(retrieved) if k is None else max(0, int(k))
    ranked = list(retrieved[:limit])
    relevant_docs = {_normalize_name(name) for name in relevant_document_names}
    relevant_ids = {str(chunk_id) for chunk_id in relevant_chunk_ids}
    total_relevant = len(relevant_docs | relevant_ids)

    def is_relevant(item: Any) -> bool:
        doc_name = _normalize_name(_value(item, "document_name", ""))
        chunk_id = str(_value(item, "chunk_id", ""))
        return (doc_name in relevant_docs) or (chunk_id in relevant_ids)

    relevant_flags = [is_relevant(item) for item in ranked]
    relevant_count = sum(relevant_flags)
    precision = relevant_count / len(ranked) if ranked else 0.0
    recall = min(1.0, relevant_count / total_relevant) if total_relevant else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if precision + recall else 0.0
    first_relevant_rank = next(
        (rank for rank, matched in enumerate(relevant_flags, start=1) if matched),
        None,
    )
    reciprocal_rank = 1.0 / first_relevant_rank if first_relevant_rank else 0.0
    scores = [float(_value(item, "score", 0.0) or 0.0) for item in ranked]

    result: Dict[str, Any] = {
        "retrieved_count": len(ranked),
        "relevant_retrieved": relevant_count,
        "relevant_expected": total_relevant,
        "precision_at_k": round(precision, 4),
        "recall_at_k": round(recall, 4),
        "f1_at_k": round(f1, 4),
        "hit_rate": 1.0 if relevant_count else 0.0,
        "reciprocal_rank": round(reciprocal_rank, 4),
        "mean_retrieval_score": round(mean(scores), 4) if scores else 0.0,
    }
    if retrieval_time_seconds is not None:
        result["retrieval_time_seconds"] = round(max(0.0, float(retrieval_time_seconds)), 6)
    return result


def benchmark_retrieval(
    retriever: Any,
    evaluation_cases: Sequence[Mapping[str, Any]],
    top_k: int = 5,
) -> Dict[str, Any]:
    """Run labelled query cases and return per-query plus aggregate metrics.

    Each case must contain query and may contain relevant_documents
    and/or relevant_chunk_ids. The retriever must expose
    retrieve_evidence(query, top_k=...).
    """
    per_query: List[Dict[str, Any]] = []
    for case in evaluation_cases:
        query = str(case.get("query", "")).strip()
        if not query:
            raise ValueError("Every retrieval evaluation case must include a non-empty query.")

        started = time.perf_counter()
        retrieved, strength = retriever.retrieve_evidence(query, top_k=top_k)
        elapsed = time.perf_counter() - started
        metrics = evaluate_retrieval(
            retrieved,
            relevant_document_names=case.get("relevant_documents", ()),
            relevant_chunk_ids=case.get("relevant_chunk_ids", ()),
            k=top_k,
            retrieval_time_seconds=elapsed,
        )
        per_query.append({
            "query": query,
            "evidence_strength": strength,
            **metrics,
        })

    metric_names = (
        "precision_at_k",
        "recall_at_k",
        "f1_at_k",
        "hit_rate",
        "reciprocal_rank",
        "retrieval_time_seconds",
    )
    aggregate = {
        name: round(mean(item[name] for item in per_query), 4) if per_query else 0.0
        for name in metric_names
    }
    return {
        "query_count": len(per_query),
        "top_k": top_k,
        "aggregate": aggregate,
        "results": per_query,
    }
