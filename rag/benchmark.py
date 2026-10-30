"""Side-by-side latency benchmark for direct LLM and retrieval-grounded answers."""
from __future__ import annotations

import time
from typing import Any, Dict


def compare_llm_only_vs_rag(query: str, retriever: Any, llm_service: Any, agent: Any, top_k: int = 5) -> Dict[str, Any]:
    """Run the same question through direct generation and the complete RAG workflow.

    This is a single-query diagnostic comparison, not a statistical performance
    claim. Retrieval relevance is measured separately when labels are supplied.
    """
    question = (query or "").strip()
    if not question:
        raise ValueError("A non-empty query is required for benchmarking.")

    llm_started = time.perf_counter()
    llm_only_answer = llm_service.generate_text_response(
        system_prompt=(
            "You are a legal research assistant. Answer the question using your general "
            "model knowledge only. Do not claim to have inspected case documents. "
            "State uncertainty where appropriate."
        ),
        user_prompt=question,
        temperature=0.1,
    )
    llm_only_seconds = time.perf_counter() - llm_started

    retrieval_started = time.perf_counter()
    retrieved_chunks, evidence_strength = retriever.retrieve_evidence(question, top_k=top_k)
    retrieval_seconds = time.perf_counter() - retrieval_started

    rag_started = time.perf_counter()
    workflow_state = agent.run_workflow(question, top_k=top_k)
    rag_seconds = time.perf_counter() - rag_started

    rag_answer = workflow_state.response or ""
    if workflow_state.structured_answer:
        rag_answer = workflow_state.structured_answer.answer

    return {
        "query": question,
        "llm_only": {
            "answer": llm_only_answer,
            "latency_seconds": round(llm_only_seconds, 4),
        },
        "rag": {
            "answer": rag_answer,
            "latency_seconds": round(rag_seconds, 4),
            "retrieval_latency_seconds": round(retrieval_seconds, 4),
            "retrieved_count": len(retrieved_chunks),
            "evidence_strength": evidence_strength,
            "source_count": len(workflow_state.sources),
            "workflow_status": workflow_state.status,
        },
        "latency_delta_seconds": round(rag_seconds - llm_only_seconds, 4),
        "measurement_note": (
            "Single-run timings are diagnostic and vary with model warm-up, hardware, "
            "provider load, and cache state. Repeat measurements before drawing conclusions."
        ),
    }
