"""Comprehensive automated test suite for deterministic timeline and witness contradiction workflows.

Covers:
- Timeline: exact timestamps, partial dates, approximate times, undated events, duplicate merging, chronological ordering, insufficient evidence.
- Contradictions: conflicting accounts, compatible accounts, non-contradictory differences, court vs testimony, missing evidence.
- Integration tests for the 5 mandated queries:
  1. Give me the timeline of events.
  2. Are there contradictions between the witness statements?
  3. What happened in this case?
  4. What evidence connects the accused to the crime?
  5. Who won the 2024 FIFA World Cup?
"""
import pytest
from typing import List
from models.schemas import DocumentChunk, IntentType, LegalResearchAnswer
from agent.timeline_pipeline import TimelinePipeline
from agent.contradiction_pipeline import ContradictionPipeline
from agent.legal_research_agent import LegalResearchAgent
from rag.retriever import RAGRetriever
from rag.vector_store import VectorStoreManager
from rag.embeddings import EmbeddingService
from llm.llm_service import LLMService


class MockEmbeddingService(EmbeddingService):
    def __init__(self):
        super().__init__(provider="local")
    def embed_texts(self, texts):
        return [[0.05] * 384 for _ in texts]


class MockVectorStoreManager(VectorStoreManager):
    def __init__(self, chunks: List[DocumentChunk]):
        self.chunks = chunks
        self.backend = "mock"
    def search(self, query, top_k=5, embedding_service=None):
        return self.chunks[:top_k]


class MockLLMService(LLMService):
    def __init__(self):
        super().__init__(provider="mock")
    def generate_structured_response(self, query, intent, retrieved_chunks, evidence_strength, normalized_query=None, analysis_notes=None):
        return self._build_fallback_answer(query, str(intent), retrieved_chunks, analysis_notes, normalized_query), "{}", ""


def get_test_case_chunks() -> List[DocumentChunk]:
    """Generate fresh synthetic case chunks for multi-document testing."""
    return [
        DocumentChunk(
            chunk_id="chunk_fir",
            document_name="01_FIR_and_Complaint.txt",
            document_type="FIR",
            page_number=1,
            score=0.92,
            text="FIR 412/2024: Accused Ramesh Kumar was charged with the murder of Vikrant Mehta occurring at the industrial warehouse on 13 November 2024. The prosecution alleges intentional homicide."
        ),
        DocumentChunk(
            chunk_id="chunk_w1",
            document_name="02_Witness_Statements.txt",
            document_type="Witness Statement",
            page_number=1,
            score=0.90,
            text="STATEMENT 01: WITNESS ANKIT SHARMA. At approximately 21:10 HRS, I was stationed near the rear service entrance gate. I saw accused Ramesh Kumar entering through the service gate wearing a dark navy blue hooded sweatshirt."
        ),
        DocumentChunk(
            chunk_id="chunk_w2",
            document_name="02_Witness_Statements.txt",
            document_type="Witness Statement",
            page_number=2,
            score=0.88,
            text="STATEMENT 02: WITNESS PRIYA NAIR. At 21:15 HRS on 13th November 2024, I claim I observed Ramesh Kumar standing outside the front main entrance wearing a bright red leather jacket near a motorcycle."
        ),
        DocumentChunk(
            chunk_id="chunk_cctv",
            document_name="03_CCTV_Analysis_Report.txt",
            document_type="CCTV Report",
            page_number=1,
            score=0.85,
            text="Camera 01 logged entry of suspect at 21:08:12 HRS through rear service door wearing navy hoodie. Camera 02 recorded suspect hurriedly exiting emergency stairwell door at 21:39:45 HRS."
        ),
        DocumentChunk(
            chunk_id="chunk_pm",
            document_name="05_Postmortem_Report.txt",
            document_type="Postmortem Report",
            page_number=2,
            score=0.82,
            text="Autopsy PM-894/2024 established estimated Time of Death between 21:15 HRS and 21:35 HRS on 13th November 2024."
        ),
        DocumentChunk(
            chunk_id="chunk_judgment",
            document_name="08_Court_Judgment.txt",
            document_type="Court Judgment",
            page_number=4,
            score=0.80,
            text="The Sessions Court pronounced judgment on 10th January 2025 convicting the accused. The court evaluated the contradiction between Ankit Sharma and Priya Nair and treated Priya Nair's observation as an honest error."
        )
    ]


# =============================================================================
# 1. TIMELINE EXTRACTION TESTS
# =============================================================================

def test_timeline_exact_timestamps_and_chronology():
    """Verify exact timestamp extraction and strict chronological ordering."""
    chunks = [
        DocumentChunk(
            chunk_id="c1", document_name="Doc1.txt", document_type="Police Report", page_number=1, score=0.9,
            text="At 21:39 HRS on 13 November 2024, the suspect exited through the rear exit."
        ),
        DocumentChunk(
            chunk_id="c2", document_name="Doc2.txt", document_type="CCTV", page_number=1, score=0.9,
            text="At 21:08 HRS on 13 November 2024, the suspect entered through the rear gate."
        ),
        DocumentChunk(
            chunk_id="c3", document_name="Doc3.txt", document_type="Witness", page_number=1, score=0.9,
            text="At 21:15 HRS on 13 November 2024, the witness saw suspect at the entrance."
        )
    ]
    res = TimelinePipeline.extract_events_structured(chunks)
    assert len(res.events) == 3
    # Check chronological ordering: 21:08 < 21:15 < 21:39
    assert res.events[0].time == "21:08 HRS"
    assert res.events[1].time == "21:15 HRS"
    assert res.events[2].time == "21:39 HRS"


def test_timeline_approximate_times_preserved():
    """Verify approximate expressions like 'around 8 PM' are preserved and not forced into false precision."""
    chunk = DocumentChunk(
        chunk_id="c_approx", document_name="Witness.txt", document_type="Witness", page_number=1, score=0.9,
        text="The security guard noted that at around 20:30 HRS on 13 November 2024, the visitor arrived."
    )
    res = TimelinePipeline.extract_events_structured([chunk])
    assert len(res.events) >= 1
    evt = res.events[0]
    assert "around" in evt.time.lower()
    assert "20:30" in evt.time


def test_timeline_partial_dates_and_undated_events():
    """Verify events with partial dates or without timestamps are captured without being discarded."""
    c_partial = DocumentChunk(
        chunk_id="cp", document_name="Brief.txt", document_type="Brief", page_number=1, score=0.9,
        text="On 10 November 2024, the suspect visited the building lobby demanding a meeting."
    )
    c_undated = DocumentChunk(
        chunk_id="cu", document_name="Police.txt", document_type="Police", page_number=1, score=0.9,
        text="The investigating officer searched the premises and seized the weapon."
    )
    res = TimelinePipeline.extract_events_structured([c_partial, c_undated])
    assert len(res.events) == 1
    assert "10 November 2024" in res.events[0].date
    assert len(res.undated_events) >= 1
    assert any("seized" in u.event_description.lower() for u in res.undated_events)


def test_timeline_duplicate_merging_across_documents():
    """Verify identical events across multiple documents are merged with combined sources."""
    c1 = DocumentChunk(
        chunk_id="cctv_01", document_name="03_CCTV.txt", document_type="CCTV", page_number=1, score=0.9,
        text="At 21:08:12 HRS on 13 November 2024, Camera 01 captured suspect entering through rear gate."
    )
    c2 = DocumentChunk(
        chunk_id="judg_01", document_name="08_Judgment.txt", document_type="Judgment", page_number=2, score=0.88,
        text="At 21:08:12 HRS on 13 November 2024, Camera 01 captured suspect entering through rear gate."
    )
    res = TimelinePipeline.extract_events_structured([c1, c2])
    assert len(res.events) == 1
    merged = res.events[0]
    assert "cctv_01" in merged.source_chunk_ids
    assert "judg_01" in merged.source_chunk_ids
    assert "03_CCTV.txt" in merged.source_document_ids
    assert "08_Judgment.txt" in merged.source_document_ids


def test_timeline_insufficient_evidence_handling():
    """Verify empty or non-timeline records return a controlled error without hanging."""
    empty_res = TimelinePipeline.extract_events_structured([])
    assert empty_res.validation_status == "insufficient"
    table = TimelinePipeline.format_timeline_markdown_table(empty_res)
    assert "Controlled Error" in table
    assert "Insufficient temporal markers" in table


# =============================================================================
# 2. CONTRADICTION ANALYSIS TESTS
# =============================================================================

def test_contradiction_conflicting_witness_accounts():
    """Verify genuinely conflicting witness claims on time, location, and attire are detected."""
    chunks = [
        DocumentChunk(
            chunk_id="w1", document_name="02_Witness.txt", document_type="Witness", page_number=1, score=0.9,
            text="Security Guard Ankit Sharma stated that accused entered rear gate at 21:08 HRS in a navy blue hoodie."
        ),
        DocumentChunk(
            chunk_id="w2", document_name="02_Witness.txt", document_type="Witness", page_number=2, score=0.88,
            text="Senior Accountant Priya Nair stated she saw accused at front gate at 21:15 HRS in a bright red jacket."
        )
    ]
    res = ContradictionPipeline.analyze_contradictions(chunks)
    assert res.contradiction_identified is True
    assert len(res.comparisons) >= 1
    comp = res.comparisons[0]
    assert "rear" in comp.nature_of_discrepancy.lower() and "front" in comp.nature_of_discrepancy.lower()
    assert "hoodie" in comp.nature_of_discrepancy.lower() and "jacket" in comp.nature_of_discrepancy.lower()
    assert comp.assessment == "Confirmed Contradiction"


def test_contradiction_compatible_accounts_no_false_manufacturing():
    """Verify compatible witness statements do not manufacture false contradictions."""
    chunks = [
        DocumentChunk(
            chunk_id="comp_w1", document_name="02_Witness.txt", document_type="Witness", page_number=1, score=0.9,
            text="Receptionist stated that victim entered the elevator at 20:30 HRS heading to 4th floor."
        ),
        DocumentChunk(
            chunk_id="comp_w2", document_name="02_Witness.txt", document_type="Witness", page_number=2, score=0.88,
            text="Janitor stated that victim was in the 4th floor corridor at 20:32 HRS unlocking his office."
        )
    ]
    res = ContradictionPipeline.analyze_contradictions(chunks)
    # Consecutive compatible sequence is NOT a contradiction
    assert res.contradiction_identified is False
    assert len(res.comparisons) == 0


def test_contradiction_distinguishes_court_finding_from_witness_testimony():
    """Verify judicial resolution as honest error does not erase the factual contradiction."""
    chunks = [
        DocumentChunk(
            chunk_id="w1", document_name="02_Witness.txt", document_type="Witness", page_number=1, score=0.9,
            text="Security Guard stated accused entered rear gate at 21:08 HRS in dark hoodie."
        ),
        DocumentChunk(
            chunk_id="w2", document_name="02_Witness.txt", document_type="Witness", page_number=2, score=0.88,
            text="Accountant stated accused was at front gate at 21:15 HRS in red jacket."
        ),
        DocumentChunk(
            chunk_id="judg", document_name="08_Judgment.txt", document_type="Judgment", page_number=3, score=0.85,
            text="The court held the accountant observation was an honest error and did not undermine the prosecution."
        )
    ]
    res = ContradictionPipeline.analyze_contradictions(chunks)
    assert res.contradiction_identified is True
    formatted = ContradictionPipeline.format_contradiction_markdown(res)
    assert "does NOT erase" in formatted or "does not negate" in formatted
    assert "honest error" in formatted.lower()


def test_contradiction_missing_witness_evidence_reports_limitation():
    """Verify that when only forensic or ballistics logs are provided, tool reports missing witness evidence."""
    non_witness_chunks = [
        DocumentChunk(
            chunk_id="f1", document_name="04_Forensic.txt", document_type="Forensic", page_number=1, score=0.9,
            text="7.65mm cartridge case matched the country-made pistol recovered from the water reservoir."
        )
    ]
    res = ContradictionPipeline.analyze_contradictions(non_witness_chunks)
    assert res.status == "insufficient_evidence"
    assert res.contradiction_identified is False
    assert len(res.limitations) >= 1
    assert "Insufficient witness evidence" in res.limitations[0]


# =============================================================================
# 3. INTEGRATION TESTS FOR THE 5 SPECIFIC USER QUERIES
# =============================================================================

def test_integration_query_1_timeline_deterministic_dispatch():
    """Integration Query 1: 'Give me the timeline of events.' must invoke timeline operation deterministically."""
    retriever = RAGRetriever(MockVectorStoreManager(get_test_case_chunks()), MockEmbeddingService())
    agent = LegalResearchAgent(retriever=retriever, llm_service=MockLLMService())

    state = agent.run_workflow(query="Give me the timeline of events.", top_k=5)

    assert state.status == "Completed"
    assert state.metrics["workflow_dispatch_status"] == "deterministic_timeline"
    assert state.metrics["required_operation"] == "timeline_extraction"
    assert state.metrics["required_operation_invoked"] is True
    assert state.metrics["required_operation_status"] == "completed"
    assert any(t.tool_name == "extract_timeline_events" for t in state.executed_tools)
    assert len(state.timeline_events) >= 3
    assert "CHRONOLOGICAL EVENT TIMELINE" in state.response or "### Chronological Event Timeline" in state.response


def test_integration_query_2_contradictions_deterministic_dispatch():
    """Integration Query 2: 'Are there contradictions between the witness statements?' must invoke contradiction operation deterministically."""
    retriever = RAGRetriever(MockVectorStoreManager(get_test_case_chunks()), MockEmbeddingService())
    agent = LegalResearchAgent(retriever=retriever, llm_service=MockLLMService())

    state = agent.run_workflow(query="Are there contradictions between the witness statements?", top_k=5)

    assert state.status == "Completed"
    assert state.metrics["workflow_dispatch_status"] == "deterministic_contradiction"
    assert state.metrics["required_operation"] == "witness_contradiction_analysis"
    assert state.metrics["required_operation_invoked"] is True
    assert state.metrics["required_operation_status"] == "completed"
    assert any(t.tool_name == "analyze_witness_contradictions" for t in state.executed_tools)
    assert state.contradiction_result is not None
    assert state.contradiction_result.contradiction_identified is True
    assert "Contradiction Identified: Yes" in state.response


def test_integration_query_3_case_summary_retained():
    """Integration Query 3: 'What happened in this case?' must retain standard case research behavior."""
    retriever = RAGRetriever(MockVectorStoreManager(get_test_case_chunks()), MockEmbeddingService())
    agent = LegalResearchAgent(retriever=retriever, llm_service=MockLLMService())

    state = agent.run_workflow(query="What happened in this case?", top_k=5)

    assert state.status == "Completed"
    assert state.metrics["workflow_dispatch_status"] == "standard_research"
    assert state.structured_answer is not None
    assert len(state.sources) > 0


def test_integration_query_4_evidence_connection_retained():
    """Integration Query 4: 'What evidence connects the accused to the crime?' must retain evidence analysis behavior."""
    retriever = RAGRetriever(MockVectorStoreManager(get_test_case_chunks()), MockEmbeddingService())
    agent = LegalResearchAgent(retriever=retriever, llm_service=MockLLMService())

    state = agent.run_workflow(query="What evidence connects the accused to the crime?", top_k=5)

    assert state.status == "Completed"
    assert state.metrics["workflow_dispatch_status"] == "standard_research"
    assert state.structured_answer is not None
    assert len(state.sources) > 0


def test_integration_query_5_fifa_world_cup_out_of_scope_refusal():
    """Integration Query 5: 'Who won the 2024 FIFA World Cup?' must refuse without hallucination, with 0 sources and insufficient evidence."""
    empty_retriever = RAGRetriever(MockVectorStoreManager([]), MockEmbeddingService())
    agent = LegalResearchAgent(retriever=empty_retriever, llm_service=MockLLMService())

    state = agent.run_workflow(query="Who won the 2024 FIFA World Cup?")

    assert state.status == "Completed"
    assert state.evidence_strength == "Insufficient"
    assert len(state.sources) == 0
    assert state.metrics["specialist_agents_completed"] == 0
    assert "could not find sufficient supporting information" in state.structured_answer.answer.lower()
