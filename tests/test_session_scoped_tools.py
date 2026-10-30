"""Regression tests for request-scoped legal tools and case-agnostic retrieval."""
from agent.tools import build_legal_tools
from rag.retriever import RAGRetriever
from models.schemas import DocumentChunk


class FakeRetriever:
    def __init__(self, label):
        self.label = label

    def retrieve_evidence(self, **kwargs):
        return [DocumentChunk(
            chunk_id=self.label,
            document_name=f"{self.label}.txt",
            document_type="Case Record",
            page_number=1,
            text=f"evidence from {self.label}",
        )], "Medium"

    def format_chunks_for_context(self, chunks):
        return f"evidence from {self.label}"


def test_tools_are_bound_to_their_own_retriever_instance():
    first = {item.name: item for item in build_legal_tools(FakeRetriever("case A"))}
    second = {item.name: item for item in build_legal_tools(FakeRetriever("case B"))}

    result_a = first["retrieve_case_evidence"].invoke({"query": "find evidence"})
    result_b = second["retrieve_case_evidence"].invoke({"query": "find evidence"})

    assert "evidence from case A" in result_a
    assert "evidence from case B" in result_b
    assert "case B" not in result_a
    assert "case A" not in result_b


def test_analysis_tools_require_explicit_request_evidence():
    tools = {item.name: item for item in build_legal_tools(FakeRetriever("isolated"))}
    summary = tools["summarize_case_facts"].invoke({"evidence_text": ""})
    timeline = tools["extract_timeline_events"].invoke({"evidence_text": ""})
    contradiction = tools["analyze_witness_contradictions"].invoke({"evidence_text": ""})

    assert "No retrieved case evidence" in summary
    assert "No retrieved evidence" in timeline
    assert "No retrieved evidence" in contradiction


def test_subqueries_do_not_inject_sample_case_names_or_locations():
    retriever = RAGRetriever.__new__(RAGRetriever)
    queries = retriever._generate_sub_queries(
        "Compare witness statements with CCTV and assess the alibi.",
        "CONTRADICTION_ANALYSIS",
    )
    joined = " ".join(queries).lower()

    for sample_detail in ("ankit sharma", "apex towers", "ramesh kumar",
                          "blue dart cafe", "aundh", "shivajinagar"):
        assert sample_detail not in joined

    assert any("witness statement" in query.lower() for query in queries)
    assert any("cctv" in query.lower() for query in queries)


def test_forensic_and_judgment_expansion_is_generic():
    retriever = RAGRetriever.__new__(RAGRetriever)
    queries = retriever._generate_sub_queries(
        "Retrieve forensic findings and final court judgment.",
        "COURT_FINDINGS",
    )
    joined = " ".join(queries).lower()

    assert "forensic laboratory findings" in joined
    assert "court judgment final order" in joined
    assert "04_forensic_lab_report" not in joined
    assert "08_court_judgment" not in joined


def test_unsupported_tool_calling_provider_is_reported_as_skipped():
    from agent.legal_research_agent import LegalResearchAgent
    from models.schemas import WorkflowState

    class GeminiService:
        provider = "gemini"

    agent = LegalResearchAgent(FakeRetriever("case"), GeminiService())
    state = WorkflowState(query="Summarize the case")
    result = agent.execute_tool_calling_loop("Summarize the case", state, "source excerpt")

    assert result == ""
    assert state.metrics["tool_loop_status"] == "skipped"
    assert "no LangChain tool-calling adapter" in state.metrics["tool_loop_reason"]
