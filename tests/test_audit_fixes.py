"""Automated verification suite for Multi-Agent CSI RAG audit fixes (Tests A through H)."""
import json
import re
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


# ─────────────────────────────────────────────────────────────────────────────
# REGRESSION TEST SUITE (Items 1 through 9)
# ─────────────────────────────────────────────────────────────────────────────

def test_1_complete_chronological_timeline_extraction():
    """Verify chronological ordering, proper sentence boundaries, and TOD window qualification."""
    full_timeline_text = (
        "--- EVIDENCE ITEM 01: 02_Witness_Statements.txt (p. 3)\n"
        "On 10th November 2024 at 17:00 HRS, Ramesh Kumar came to the building lobby demanding to meet Mr. Malhotra and shouted verbal threats.\n\n"
        "--- EVIDENCE ITEM 02: 02_Witness_Statements.txt (p. 1)\n"
        "On 13th November 2024, at around 20:30 HRS, Mr. Vikram Malhotra arrived in his car and went up to his 4th floor office #402.\n\n"
        "--- EVIDENCE ITEM 03: 03_CCTV_Analysis_Report.txt (p. 1)\n"
        "At 21:08:12 HRS on 13th November 2024, Camera 01 captured suspect Ramesh Kumar entering through the rear service gate in a navy blue hooded sweatshirt.\n\n"
        "--- EVIDENCE ITEM 04: 02_Witness_Statements.txt (p. 2)\n"
        "At 21:15 HRS on 13th November 2024, Priya Nair claims she observed Ramesh Kumar standing outside the front main entrance in a red leather jacket.\n\n"
        "--- EVIDENCE ITEM 05: 05_Postmortem_Report.txt (p. 2)\n"
        "Autopsy PM-894/2024 established estimated Time of Death between 21:15 HRS and 21:35 HRS on 13th November 2024.\n\n"
        "--- EVIDENCE ITEM 06: 03_CCTV_Analysis_Report.txt (p. 1)\n"
        "At 21:39:45 HRS on 13th November 2024, suspect was captured hurriedly exiting the rear service emergency stairwell door carrying a black leather briefcase.\n\n"
        "--- EVIDENCE ITEM 07: 08_Court_Judgment.txt (p. 5)\n"
        "On 10th January 2025, the Sessions Court pronounced judgment convicting accused Ramesh Kumar under IPC 302 and 392."
    )
    tools = {t.name: t for t in build_legal_tools(None)}
    result = tools["extract_timeline_events"].invoke({"evidence_text": full_timeline_text})

    assert "CHRONOLOGICAL EVENT TIMELINE" in result
    idx_10_nov = result.find("10th November 2024")
    idx_21_08 = result.find("21:08")
    idx_21_15 = result.find("21:15")
    idx_21_39 = result.find("21:39")
    idx_10_jan = result.find("10th January 2025")

    assert idx_10_nov != -1 and idx_21_08 != -1 and idx_21_15 != -1 and idx_21_39 != -1 and idx_10_jan != -1
    assert idx_10_nov < idx_21_08 < idx_21_15 < idx_21_39 < idx_10_jan

    # Verify event categorization
    assert "[Witness Observation]" in result
    assert "[CCTV Event]" in result
    assert "[Judicial Event]" in result

    # Verify TOD qualification
    assert "[Forensic Estimate: Estimated Time Window of Death]" in result
    assert "Medical estimate window; not an exact observed event time" in result

    # Verify no sliced sentence fragments like "a ("
    assert not re.search(r'\-\s*\*\*[^*]+\*\*\s*:\s*[a-z]\s*\(', result)


def test_2_timeline_extraction_from_multiple_chunks_in_one_document():
    """Verify extracting timeline events across multiple chunks originating from one judgment document."""
    multi_chunks = [
        DocumentChunk(
            chunk_id="cj_c1",
            document_name="08_Court_Judgment.txt",
            document_type="Court Judgment",
            page_number=1,
            score=0.9,
            text="1. Prosecution Case: Offence occurred on 13th November 2024 between 21:15 HRS and 21:35 HRS. Accused entered rear gate at 21:08 HRS."
        ),
        DocumentChunk(
            chunk_id="cj_c2",
            document_name="08_Court_Judgment.txt",
            document_type="Court Judgment",
            page_number=3,
            score=0.88,
            text="3. Contradictions: Guard Ankit observed accused at 21:08 HRS, while Priya Nair claimed seeing accused at front gate at 21:15 HRS."
        ),
        DocumentChunk(
            chunk_id="cj_c3",
            document_name="08_Court_Judgment.txt",
            document_type="Court Judgment",
            page_number=4,
            score=0.87,
            text="4. Forensic Evidence: Time of Death was between 21:15 HRS and 21:35 HRS. Accused exited at 21:39 HRS. Judgment pronounced on 10th January 2025."
        )
    ]
    retriever = RAGRetriever(MockVectorStoreManager(multi_chunks), MockEmbeddingService())
    tools = {t.name: t for t in build_legal_tools(retriever)}
    context = retriever.format_chunks_for_context(multi_chunks)

    result = tools["extract_timeline_events"].invoke({"evidence_text": context})
    assert "21:08" in result
    assert "21:15" in result
    assert "21:39" in result
    assert "10th January 2025" in result


def test_3_malformed_timeline_tool_output_rejection():
    """Verify tool returns a controlled error when evidence has insufficient temporal markers."""
    tools = {t.name: t for t in build_legal_tools(None)}
    no_timestamps_text = (
        "The accused pleaded an alibi under Section 11 of the Indian Evidence Act. "
        "The court analyzed legal arguments, statutory precedents, and evidentiary burdens."
    )
    result = tools["extract_timeline_events"].invoke({"evidence_text": no_timestamps_text})
    assert "Controlled Error" in result
    assert "Insufficient temporal markers" in result


def test_4_contradiction_detection_between_witness_accounts():
    """Verify witness accounts are compared across timing, location, attire, and distinguished from court resolution."""
    tools = {t.name: t for t in build_legal_tools(None)}
    witness_text = (
        "--- 02_Witness_Statements.txt\n"
        "Security Guard Ankit Sharma stated accused entered through rear gate at 21:08 HRS wearing a navy blue hoodie.\n"
        "Senior Accountant Priya Nair stated she saw accused standing at front gate at 21:15 HRS wearing a bright red leather jacket.\n"
        "--- 08_Court_Judgment.txt\n"
        "CCTV Cameras 01 and 02 corroborated Ankit Sharma's account and attire. Priya Nair's observation was treated as an honest error in timing or identity."
    )
    result = tools["analyze_witness_contradictions"].invoke({"evidence_text": witness_text})

    assert "Contradiction Identified: Yes" in result
    assert "Security Guard Ankit Sharma" in result
    assert "Senior Accountant Priya Nair" in result
    assert "21:08" in result and "21:15" in result
    assert "rear" in result.lower() and "front" in result.lower()
    assert "hoodie" in result.lower() and "jacket" in result.lower()
    assert "CCTV" in result
    assert "honest error" in result.lower()
    assert "does not erase" in result.lower() or "does not negate" in result.lower()


def test_5_specialist_disagreement_reconciliation_and_correction():
    """Verify coordinator corrects a specialist that erroneously denies contradiction due to court resolution."""
    class StubDenialLLM(LLMService):
        def __init__(self):
            super().__init__(provider="mock")
        def generate_text_response(self, system_prompt, user_prompt, temperature=0.1):
            return "Based on the court judgment, there is no contradiction between the witness statements because the court held it was an honest error."

    coordinator = MultiAgentCoordinator(StubDenialLLM())
    evidence_ctx = (
        "Security Guard Ankit Sharma (rear gate, 21:08, navy blue hoodie). "
        "Senior Accountant Priya Nair (front gate, 21:15, red leather jacket). "
        "Court resolved discrepancy as an honest error."
    )
    analyses = coordinator.analyze(
        query="Are there contradictions between the witness statements?",
        intent=IntentType.CONTRADICTION_ANALYSIS,
        evidence_context=evidence_ctx
    )

    # The specialist output must be sanitized and must NOT state 'no contradiction'
    for a in analyses:
        assert "no contradiction between the witness statements" not in a.analysis.lower()
        assert "Contradiction Identified: Yes" in a.analysis

    reconciled = coordinator.reconcile_analyses(analyses, SAMPLE_WITNESS_CHUNKS, IntentType.CONTRADICTION_ANALYSIS)
    assert "Authoritative Consensus" in reconciled
    assert "factual contradiction DOES exist" in reconciled


def test_6_answer_and_limitations_consistency():
    """Verify that contradictory limitations are scrubbed and generic limitations removed."""
    answer = LegalResearchAnswer(
        query="Are there contradictions between the witness statements?",
        intent="CONTRADICTION_ANALYSIS",
        answer="Contradiction identified: Yes. Factual discrepancies exist between Ankit Sharma and Priya Nair.",
        key_findings=[
            "Contradiction identified: Ankit Sharma and Priya Nair gave factually conflicting accounts.",
            "Accused was convicted under IPC 302 and 392."
        ],
        evidence_summary=["[08_Court_Judgment.txt, p. 3] Evaluated witness contradictions."],
        limitations=[
            "The judgment contains no direct contradictions between the witness statements.",
            "No explicit court verdict or operative order was present.",
            "Information strictly bounded by uploaded case documents.",
            "Original audio recordings of depositions were not available."
        ],
        sources=[]
    )

    reconciled_answer = LLMService._validate_and_reconcile_answer_consistency(
        answer,
        query="Are there contradictions between the witness statements?",
        chunks=SAMPLE_WITNESS_CHUNKS
    )

    lims = [l.lower() for l in reconciled_answer.limitations]
    assert not any("no direct contradictions" in l for l in lims)
    assert not any("no explicit court verdict" in l for l in lims)
    assert not any("strictly bounded" in l for l in lims)
    assert any("original audio recordings" in l for l in lims)


def test_7_correct_evidence_selection_for_each_intent():
    """Verify select_evidence_for_intent filters chunks properly for each intent type."""
    from agent.legal_research_agent import select_evidence_for_intent

    all_chunks = [
        DocumentChunk(chunk_id="t1", document_name="03_CCTV.txt", document_type="CCTV", page_number=1, score=0.8,
                      text="Timestamp 21:08:12 HRS on 13 November 2024: Rear gate entry recorded on CCTV."),
        DocumentChunk(chunk_id="w1", document_name="02_Witness.txt", document_type="Witness", page_number=1, score=0.8,
                      text="Witness statement: Ankit Sharma deposed that accused entered through rear gate."),
        DocumentChunk(chunk_id="f1", document_name="04_Forensic.txt", document_type="Forensic", page_number=1, score=0.8,
                      text="Ballistics and GSR test confirmed gunshot residue on accused right hand."),
        DocumentChunk(chunk_id="a1", document_name="06_Police.txt", document_type="Police", page_number=1, score=0.8,
                      text="Alibi verification: Cell tower Shivajinagar-BS-04 CDR log refutes presence at cafe.")
    ]

    timeline_selected = select_evidence_for_intent(all_chunks, IntentType.TIMELINE, "timeline of events")
    assert any(c.chunk_id == "t1" for c in timeline_selected)

    witness_selected = select_evidence_for_intent(all_chunks, IntentType.CONTRADICTION_DETECTION, "witness contradictions")
    assert any(c.chunk_id == "w1" for c in witness_selected)

    forensic_selected = select_evidence_for_intent(all_chunks, IntentType.EVIDENCE_ANALYSIS, "forensic findings")
    assert any(c.chunk_id == "f1" for c in forensic_selected)

    alibi_selected = select_evidence_for_intent(all_chunks, IntentType.ACCUSED_INFORMATION, "alibi verification")
    assert any(c.chunk_id == "a1" for c in alibi_selected)


def test_8_citation_integrity_and_source_deduplication():
    """Verify 5 chunks from 1 document are not counted as 5 independent documents."""
    five_chunks_same_doc = [
        DocumentChunk(chunk_id=f"cj_{i}", document_name="08_Court_Judgment.txt", document_type="Court Judgment",
                      page_number=i, score=0.9 - i * 0.02, text=f"Judgment excerpt section {i}")
        for i in range(1, 6)
    ]
    agent = LegalResearchAgent(
        retriever=RAGRetriever(MockVectorStoreManager(five_chunks_same_doc), MockEmbeddingService()),
        llm_service=MockLLMService()
    )

    state = agent.run_workflow(query="What was the court judgment?", top_k=5)
    assert state.status == "Completed"
    assert state.metrics["retrieved_chunk_count"] == 5
    assert state.metrics["distinct_document_count"] == 1


def test_9_out_of_scope_query_refusal():
    """Verify out-of-scope query refusal without hallucinations."""
    empty_retriever = RAGRetriever(MockVectorStoreManager([]), MockEmbeddingService())
    agent = LegalResearchAgent(retriever=empty_retriever, llm_service=MockLLMService())

    state = agent.run_workflow(query="Who won the 2024 FIFA World Cup?")
    assert state.status == "Completed"
    assert state.evidence_strength == "Insufficient"
    assert "could not find sufficient supporting information" in state.structured_answer.answer.lower()
    assert len(state.sources) == 0
