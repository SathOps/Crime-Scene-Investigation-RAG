"""Automated verification suite for Multi-Agent CSI RAG audit fixes (Tests A through H)."""
import json
import pytest
from agent.tools import build_legal_tools
from agent.specialist_agents import (
    MultiAgentCoordinator,
    AgentAnalysis,
    WitnessContradictionAgent,
    TimelineAnalysisAgent,
)
from agent.legal_research_agent import (
    LegalResearchAgent,
    is_standalone_hypothetical_query,
    build_hypothetical_answer,
)
from rag.retriever import RAGRetriever
from rag.vector_store import VectorStoreManager
from rag.embeddings import EmbeddingService
from models.schemas import DocumentChunk, IntentType, LegalResearchAnswer, WorkflowState
from llm.llm_service import LLMService


class MockEmbeddingService(EmbeddingService):
    """Deterministic mock embedding service."""
    def __init__(self):
        super().__init__(provider="local")

    def embed_texts(self, texts):
        return [[0.1] * 384 for _ in texts]

    def embed_query(self, query):
        return [0.1] * 384


class MockVectorStoreManager(VectorStoreManager):
    """Deterministic mock vector store returning predefined chunks."""
    def __init__(self, chunks=None):
        self.preset_chunks = chunks or []

    def search(self, query, top_k=5, embedding_service=None):
        return self.preset_chunks[:top_k]

    def get_stats(self):
        return {"total_chunks": len(self.preset_chunks), "total_documents": 2}


class MockLLMService(LLMService):
    """Deterministic mock LLM service returning structured JSON."""
    def __init__(self, answer_text="Default mock answer", provider="local"):
        super().__init__(provider=provider)
        self.answer_text = answer_text
        self.calls = []

    def generate_text_response(self, system_prompt, user_prompt, temperature=0.1):
        self.calls.append((system_prompt, user_prompt))
        return self.answer_text

    def generate_structured_response(self, query, intent, retrieved_chunks, evidence_strength, normalized_query=None, analysis_notes=""):
        return self._build_fallback_answer(query, str(intent), retrieved_chunks, analysis_notes, normalized_query), "{}", ""


# Sample chunks from actual case records
SAMPLE_WITNESS_CHUNKS = [
    DocumentChunk(
        chunk_id="ws_01",
        document_name="02_Witness_Statements.txt",
        document_type="Witness Statement",
        page_number=1,
        score=0.75,
        text=(
            "STATEMENT 01: WITNESS ANKIT SHARMA. At approximately 21:10 HRS, I was stationed near "
            "the rear service entrance gate. I saw a man walking in rapidly through the service gate. "
            "I recognized him as Mr. Ramesh Kumar. He was wearing a dark navy blue hooded sweatshirt "
            "with the hood pulled over his head, dark jeans, and black sneakers. At 21:40 HRS, I saw "
            "Ramesh Kumar running out of the rear emergency exit door holding a black leather briefcase."
        )
    ),
    DocumentChunk(
        chunk_id="ws_02",
        document_name="02_Witness_Statements.txt",
        document_type="Witness Statement",
        page_number=2,
        score=0.72,
        text=(
            "STATEMENT 02: WITNESS PRIYA NAIR. On the night of 13th November 2024, while waiting "
            "near the bus stop at 21:15 HRS, I claim I saw Ramesh Kumar standing outside the front "
            "main entrance of Apex Towers. Crucially, I observed him wearing a bright red leather jacket "
            "and dark blue jeans, standing near a red motorcycle. He appeared to be talking on his mobile phone."
        )
    ),
    DocumentChunk(
        chunk_id="cj_03",
        document_name="08_Court_Judgment.txt",
        document_type="Court Judgment",
        page_number=3,
        score=0.88,
        text=(
            "3. Evaluation of Witness Statements & Contradictions: Court examined the contradiction between "
            "Security Guard Ankit Sharma (stating accused entered rear gate at 21:08 HRS in a navy blue hoodie) "
            "and Senior Accountant Priya Nair (stating she saw accused at front gate at 21:15 HRS in a red jacket). "
            "Court Finding: CCTV Camera 01 and Camera 02 conclusive video logs corroborate Security Guard "
            "Ankit Sharma's timeline and hoodie description. Priya Nair's observation at front gate was unsupported "
            "by video evidence and constitutes an honest error in timing or identity. Minor discrepancy does not "
            "shake the core prosecution narrative."
        )
    )
]

SAMPLE_TIMELINE_CHUNKS = [
    DocumentChunk(
        chunk_id="cj_01",
        document_name="08_Court_Judgment.txt",
        document_type="Court Judgment",
        page_number=1,
        score=0.68,
        text=(
            "1. Prosecution Case Summary: Murder occurred on 13th November 2024 between 21:15 HRS and 21:35 HRS "
            "in Office #402, Apex Towers, Shivajinagar. Accused entered through rear gate at 21:08 HRS. "
            "Postmortem report established Time of Death between 21:15 HRS and 21:35 HRS. "
            "CCTV Camera 02 recorded accused exiting at 21:39 HRS. Judgment pronounced on 10th January 2025."
        )
    )
]

SAMPLE_IRRELEVANT_CHUNKS = [
    DocumentChunk(
        chunk_id="alibi_01",
        document_name="06_Police_Investigation_Report.txt",
        document_type="Police Report",
        page_number=4,
        score=0.45,
        text=(
            "Cell Tower CDR verification: Shivajinagar-BS-04 base station logs. Accused claimed presence "
            "at Blue Dart Cafe in Aundh. Testimony of cafe manager confirms accused was not present. "
            "CCTV gait analysis revealed 87.4% match."
        )
    )
]


# ─────────────────────────────────────────────────────────────────────────────
# TEST A: Witness Contradiction Detection
# ─────────────────────────────────────────────────────────────────────────────
def test_a_witness_contradiction_detection():
    """Verify that contradiction analysis correctly identifies discrepancies and distinguishes them from court resolution."""
    retriever = RAGRetriever(
        MockVectorStoreManager(SAMPLE_WITNESS_CHUNKS),
        MockEmbeddingService()
    )
    tools = {t.name: t for t in build_legal_tools(retriever)}
    
    evidence_text = retriever.format_chunks_for_context(SAMPLE_WITNESS_CHUNKS)
    result = tools["analyze_witness_contradictions"].invoke({"evidence_text": evidence_text})

    # 1. Contradiction identified must be Yes
    assert "Contradiction Identified: Yes" in result
    # 2. Time differences (21:08/21:10 vs 21:15)
    assert "21:08" in result or "21:10" in result
    assert "21:15" in result
    # 3. Location differences (rear vs front)
    assert "rear" in result.lower()
    assert "front" in result.lower()
    # 4. Clothing differences (navy blue hoodie vs red leather jacket)
    assert "hoodie" in result.lower()
    assert "jacket" in result.lower()
    # 5. Independent corroboration (CCTV Camera 01 / 02)
    assert "CCTV" in result
    # 6. Judicial resolution distinguished from factual contradiction
    assert "honest error" in result.lower()
    assert "does not erase" in result.lower() or "does not negate" in result.lower()
    # 7. Must NEVER claim no contradiction exists
    assert "no contradiction between the witness statements" not in result.lower()


# ─────────────────────────────────────────────────────────────────────────────
# TEST B: Irrelevant Evidence Rejection
# ─────────────────────────────────────────────────────────────────────────────
def test_b_irrelevant_evidence_rejection():
    """Verify that when contradiction tool receives only irrelevant alibi/cell tower evidence, it rejects it."""
    # Retriever with NO witness statements in the vector store
    empty_witness_retriever = RAGRetriever(
        MockVectorStoreManager(SAMPLE_IRRELEVANT_CHUNKS),
        MockEmbeddingService()
    )
    tools = {t.name: t for t in build_legal_tools(empty_witness_retriever)}

    irrelevant_text = empty_witness_retriever.format_chunks_for_context(SAMPLE_IRRELEVANT_CHUNKS)
    result = tools["analyze_witness_contradictions"].invoke({"evidence_text": irrelevant_text})

    # Must report insufficient witness evidence rather than fabricating
    assert "Insufficient witness evidence" in result
    assert "Contradiction Identified: Yes" not in result


# ─────────────────────────────────────────────────────────────────────────────
# TEST C: Timeline Generation & Sufficiency
# ─────────────────────────────────────────────────────────────────────────────
def test_c_timeline_generation_and_sufficiency():
    """Verify that 'Give me the timeline of events' passes sufficiency and extracts chronological timestamps."""
    retriever = RAGRetriever(
        MockVectorStoreManager(SAMPLE_TIMELINE_CHUNKS),
        MockEmbeddingService()
    )
    
    # Sufficiency check on timeline query
    chunks, strength = retriever.retrieve_evidence(
        query="Give me the timeline of events.",
        top_k=5,
        intent="TIMELINE"
    )
    
    assert len(chunks) > 0
    assert strength != "Insufficient"
    assert strength in {"High", "Medium", "Low"}

    # Timeline extraction tool verification
    tools = {t.name: t for t in build_legal_tools(retriever)}
    context = retriever.format_chunks_for_context(chunks)
    timeline_result = tools["extract_timeline_events"].invoke({"evidence_text": context})

    assert "CHRONOLOGICAL EVENT TIMELINE" in timeline_result
    assert "21:08" in timeline_result
    assert "21:15" in timeline_result
    assert "21:39" in timeline_result
    assert "13th November 2024" in timeline_result or "10th January 2025" in timeline_result


# ─────────────────────────────────────────────────────────────────────────────
# TEST D: Specialist Agent Disagreement Reconciliation
# ─────────────────────────────────────────────────────────────────────────────
def test_d_specialist_agent_disagreement_reconciliation():
    """Verify that MultiAgentCoordinator reconciles conflicting conclusions using authoritative case text."""
    coordinator = MultiAgentCoordinator(MockLLMService())
    
    conflicting_results = [
        AgentAnalysis(
            agent_name="Witness & Contradiction Agent",
            status="completed",
            analysis="Ankit Sharma saw accused at rear gate in navy hoodie at 21:08. Priya Nair saw accused at front gate in red jacket at 21:15. There are discrepancies."
        ),
        AgentAnalysis(
            agent_name="Case Facts Agent",
            status="completed",
            analysis="The court found the accused guilty and resolved the matters. No contradiction undermines the outcome."
        )
    ]

    reconciled = coordinator.reconcile_analyses(
        results=conflicting_results,
        chunks=SAMPLE_WITNESS_CHUNKS,
        intent=IntentType.CONTRADICTION_ANALYSIS
    )

    assert "Authoritative Consensus" in reconciled
    assert "factual contradiction DOES exist" in reconciled
    assert "honest error" in reconciled.lower()
    assert "CCTV Cameras 01 & 02" in reconciled


# ─────────────────────────────────────────────────────────────────────────────
# TEST E: Out-of-Scope / Hypothetical Refusal
# ─────────────────────────────────────────────────────────────────────────────
def test_e_out_of_scope_and_hypothetical_refusal():
    """Verify out-of-scope rejection and hypothetical scenario isolation."""
    # 1. Out-of-scope query
    retriever = RAGRetriever(
        MockVectorStoreManager([]),
        MockEmbeddingService()
    )
    chunks, strength = retriever.retrieve_evidence("What is the capital of France?")
    assert strength == "Insufficient"
    assert len(chunks) == 0

    # 2. Hypothetical scenario query
    hypo_query = (
        "A person is accused of theft. One witness says the incident occurred at 8 PM, "
        "while another says it occurred at 9 PM. CCTV footage shows the accused entering "
        "the building at 8:30 PM. Analyze the timeline and identify the contradiction."
    )
    assert is_standalone_hypothetical_query(hypo_query) is True
    
    hypo_answer = build_hypothetical_answer(hypo_query, hypo_query, "TIMELINE")
    assert hypo_answer is not None
    assert "hypothetical" in hypo_answer.answer.lower()
    assert len(hypo_answer.sources) == 0


# ─────────────────────────────────────────────────────────────────────────────
# TEST F: Citation Integrity
# ─────────────────────────────────────────────────────────────────────────────
def test_f_citation_integrity():
    """Verify that cited sources strictly match retrieved chunks with document name, page, and chunk_id."""
    agent = LegalResearchAgent(
        retriever=RAGRetriever(MockVectorStoreManager(SAMPLE_WITNESS_CHUNKS), MockEmbeddingService()),
        llm_service=MockLLMService()
    )

    state = agent.run_workflow(
        query="What did the witnesses observe?",
        top_k=3
    )

    assert state.status == "Completed"
    assert len(state.sources) > 0
    retrieved_chunk_ids = {c.chunk_id for c in state.retrieved_chunks}

    for src in state.sources:
        assert src["chunk_id"] in retrieved_chunk_ids
        assert src["document_name"] in {"02_Witness_Statements.txt", "08_Court_Judgment.txt"}
        assert src["page_number"] is not None
        assert src["excerpt"] != ""


# ─────────────────────────────────────────────────────────────────────────────
# TEST G: Session Isolation
# ─────────────────────────────────────────────────────────────────────────────
def test_g_session_isolation():
    """Verify that tools and agents bound to different cases do not bleed evidence."""
    retriever_case_a = RAGRetriever(MockVectorStoreManager(SAMPLE_WITNESS_CHUNKS), MockEmbeddingService())
    retriever_case_b = RAGRetriever(MockVectorStoreManager(SAMPLE_TIMELINE_CHUNKS), MockEmbeddingService())

    tools_a = {t.name: t for t in build_legal_tools(retriever_case_a)}
    tools_b = {t.name: t for t in build_legal_tools(retriever_case_b)}

    result_a = tools_a["retrieve_case_evidence"].invoke({"query": "witnesses"})
    result_b = tools_b["retrieve_case_evidence"].invoke({"query": "witnesses"})

    assert "02_Witness_Statements.txt" in result_a
    assert "02_Witness_Statements.txt" not in result_b


# ─────────────────────────────────────────────────────────────────────────────
# TEST H: Workflow Execution Status and Diagnostics
# ─────────────────────────────────────────────────────────────────────────────
def test_h_workflow_execution_status_and_diagnostics():
    """Verify accurate stage statuses, distinct document counts, and sufficiency gate behavior."""
    agent = LegalResearchAgent(
        retriever=RAGRetriever(MockVectorStoreManager(SAMPLE_WITNESS_CHUNKS), MockEmbeddingService()),
        llm_service=MockLLMService()
    )

    state = agent.run_workflow(
        query="Are there contradictions between the witness statements?",
        top_k=3
    )

    assert state.status == "Completed"
    # All executed steps must have valid statuses
    for step in state.steps:
        assert step.status in {"completed", "skipped", "failed", "in_progress"}

    # Metrics validation
    metrics = state.metrics
    assert "total_duration_seconds" in metrics
    assert "retrieved_chunk_count" in metrics
    assert metrics["retrieved_chunk_count"] == len(state.retrieved_chunks)
    assert "distinct_document_count" in metrics
    assert metrics["distinct_document_count"] == 2  # 02_Witness_Statements.txt and 08_Court_Judgment.txt
