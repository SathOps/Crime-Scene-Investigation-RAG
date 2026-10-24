"""Unit tests for specialist-agent routing and failure-safe delegation."""
from agent.specialist_agents import (
    AgentAnalysis,
    MultiAgentCoordinator,
    TimelineAnalysisAgent,
    WitnessContradictionAgent,
    CaseFactsAgent,
    EvidenceAssessmentAgent,
)
from models.schemas import IntentType
from agent.legal_research_agent import (
    is_standalone_hypothetical_query,
    build_hypothetical_answer,
)


class FakeLLMService:
    def __init__(self, response="analysis from mock", fail=False):
        self.response = response
        self.fail = fail
        self.calls = []

    def generate_text_response(self, system_prompt, user_prompt, temperature=0.1):
        self.calls.append((system_prompt, user_prompt, temperature))
        if self.fail:
            raise RuntimeError("mock provider unavailable")
        return self.response


def test_timeline_intent_selects_timeline_and_case_facts_agents():
    coordinator = MultiAgentCoordinator(FakeLLMService())
    names = [agent.agent_name for agent in coordinator.select_agents(IntentType.TIMELINE)]
    assert names == ["Timeline Analysis Agent", "Case Facts Agent"]


def test_contradiction_intent_selects_witness_and_timeline_agents():
    coordinator = MultiAgentCoordinator(FakeLLMService())
    names = [
        agent.agent_name
        for agent in coordinator.select_agents(IntentType.CONTRADICTION_DETECTION)
    ]
    assert names == ["Witness & Contradiction Agent", "Timeline Analysis Agent"]


def test_analysis_runs_each_selected_specialist_on_supplied_evidence():
    llm = FakeLLMService("Analysis grounded in supplied excerpts.")
    coordinator = MultiAgentCoordinator(llm)

    results = coordinator.analyze(
        query="Compare the witness statements with CCTV.",
        intent=IntentType.CONTRADICTION_ANALYSIS,
        evidence_context="[Source: CCTV report, page 2] Exit logged at 21:40 HRS.",
    )

    assert len(results) == 2
    assert all(result.status == "completed" for result in results)
    assert len(llm.calls) == 2
    assert all("CCTV report" in call[1] for call in llm.calls)
    assert "specialist analysis" in coordinator.format_successful_analyses(results)


def test_specialist_skips_when_no_evidence_is_supplied():
    llm = FakeLLMService()
    result = CaseFactsAgent(llm).run("Summarize the case", "CASE_SUMMARY", "")

    assert result.status == "skipped"
    assert not llm.calls


def test_specialist_failure_is_reported_without_raising():
    llm = FakeLLMService(fail=True)
    result = EvidenceAssessmentAgent(llm).run(
        "Assess this evidence", "EVIDENCE_ANALYSIS", "Source: report, page 1."
    )

    assert result.status == "failed"
    assert "mock provider unavailable" in result.error


def test_failed_analyses_are_not_included_in_synthesis_context():
    context = MultiAgentCoordinator.format_successful_analyses([
        AgentAnalysis("Case Facts Agent", "completed", "Supported fact."),
        AgentAnalysis("Timeline Analysis Agent", "failed", error="timeout"),
    ])

    assert "Supported fact." in context
    assert "timeout" not in context


def test_deterministic_local_provider_skips_llm_specialists():
    class LocalFallbackService:
        provider = "local"

        def generate_text_response(self, **kwargs):
            raise AssertionError("The deterministic fallback must not call an LLM.")

    coordinator = MultiAgentCoordinator(LocalFallbackService())
    results = coordinator.analyze(
        query="Summarize the case.",
        intent=IntentType.CASE_SUMMARY,
        evidence_context="Source: FIR, page 1.",
    )

    assert len(results) == 2
    assert all(result.status == "skipped" for result in results)



def test_generic_theft_timeline_is_detected_as_standalone_hypothetical():
    query = (
        "A person is accused of theft. One witness says the incident occurred at 8 PM, "
        "while another says it occurred at 9 PM. CCTV footage shows the accused entering "
        "the building at 8:30 PM. Analyze the timeline, identify the contradiction, "
        "summarize the relevant facts, and assess the evidence."
    )
    assert is_standalone_hypothetical_query(query)


def test_hypothetical_answer_uses_only_user_supplied_times():
    query = (
        "A person is accused of theft. One witness says the incident occurred at 8 PM, "
        "while another says it occurred at 9 PM. CCTV footage shows the accused entering "
        "the building at 8:30 PM."
    )
    answer = build_hypothetical_answer(query, query, "TIMELINE")
    assert "8:00 PM" in answer.answer
    assert "9:00 PM" in answer.answer
    assert "8:30 PM" in answer.answer
    assert "Ramesh Kumar" not in answer.answer
    assert "murder" not in answer.answer.lower()
    assert answer.sources == []
    assert any("No case documents were used" in item for item in answer.limitations)


def test_case_file_question_is_not_misclassified_as_generic_hypothetical():
    query = "Analyze the timeline in the uploaded case file and compare the CCTV with witness statements."
    assert not is_standalone_hypothetical_query(query)


def test_hypothetical_timeline_preserves_clock_times_without_false_conversion():
    answer = build_hypothetical_answer(
        "One witness says 8 PM, another says 9 PM, and CCTV shows entry at 8:30 PM.",
        "One witness says 8 PM, another says 9 PM, and CCTV shows entry at 8:30 PM.",
        "TIMELINE",
    )
    assert "8:04 PM" not in answer.answer
    assert "21:08" not in answer.answer
    assert "8:00 PM" in answer.answer
    assert "9:00 PM" in answer.answer



def test_hypothetical_workflow_reports_progress_and_skips_retrieval():
    from agent.legal_research_agent import LegalResearchAgent

    class LocalLLM:
        provider = "local"

        def generate_text_response(self, **kwargs):
            raise AssertionError("Hypothetical flow must not call the LLM.")

    class NoRetrieval:
        def retrieve_evidence(self, *args, **kwargs):
            raise AssertionError("Hypothetical flow must skip retrieval.")

    progress = []
    agent = LegalResearchAgent(NoRetrieval(), LocalLLM())
    state = agent.run_workflow(
        "A person is accused of theft. One witness says the incident occurred at 8 PM, "
        "while another says it occurred at 9 PM. CCTV footage shows the accused entering "
        "the building at 8:30 PM.",
        progress_callback=progress.append,
    )

    assert state.status == "Completed"
    assert state.sources == []
    assert any("skipping retrieval" in item.lower() for item in progress)
    assert "Ramesh Kumar" not in state.response



def test_query_normalizer_does_not_duplicate_names_or_cctv_footage():
    from utils.query_normalizer import QueryNormalizer

    query = (
        "Analyze the case of State vs. Ramesh Kumar using the uploaded case documents. "
        "Compare CCTV footage with witness statements and identify contradictions."
    )
    normalized = QueryNormalizer.normalize(query).normalized_query
    assert "Ramesh Kumar Kumar" not in normalized
    assert "CCTV footage footage" not in normalized
    assert "Ramesh Kumar" in normalized


def test_short_model_answer_is_expanded_with_attributed_evidence_and_limits():
    from llm.llm_service import LLMService
    from models.schemas import DocumentChunk

    service = LLMService(provider="local")
    chunks = [
        DocumentChunk(
            chunk_id="witness_p1",
            document_name="02_Witness_Statements.txt",
            document_type="Witness Statement",
            page_number=1,
            text="At 21:25 HRS, the guard heard a bang. At 21:40 HRS, he saw a person leaving with a briefcase.",
        ),
        DocumentChunk(
            chunk_id="judgment_p1",
            document_name="08_Court_Judgment.txt",
            document_type="Court Judgment / Order",
            page_number=1,
            text="The prosecution alleged that the accused committed the offence.",
        ),
    ]
    expanded = service._expand_short_answer("The prosecution alleges murder.", "Summarize and assess.", chunks)
    assert "[02_Witness_Statements.txt, page 1" in expanded
    assert "21:25 HRS" in expanded
    assert "does not establish the operative verdict" in expanded
    assert "proven" in expanded



def test_court_judgment_request_is_classified_as_court_findings():
    from utils.intent_classifier import IntentClassifier
    from models.schemas import IntentType

    intent, confidence = IntentClassifier.classify(
        "Retrieve the forensic findings and final court judgment. Do not infer a verdict from the charge sheet."
    )
    assert intent == IntentType.COURT_FINDINGS
    assert confidence >= 0.9


def test_forensic_and_judgment_short_answer_states_missing_results_explicitly():
    from llm.llm_service import LLMService
    from models.schemas import DocumentChunk

    service = LLMService(provider="local")
    chunks = [
        DocumentChunk(
            chunk_id="forensic_header",
            document_name="04_Forensic_Lab_Report.txt",
            document_type="Forensic / Medical Report",
            page_number=1,
            text="STATE FORENSIC SCIENCE LABORATORY. Evidence Item-A: spent shell casing. LABORATORY FINDINGS:",
        ),
        DocumentChunk(
            chunk_id="judgment_header",
            document_name="08_Court_Judgment.txt",
            document_type="Court Judgment / Order",
            page_number=1,
            text="JUDGMENT & FINAL ORDER: Prosecution alleged that the accused committed murder.",
        ),
    ]
    expanded = service._expand_short_answer(
        "The court confirms the accused is innocent and forensic evidence supports his alibi.",
        "Summarize forensic findings and the final court judgment.",
        chunks,
    )
    assert "does not establish the operative verdict" in expanded
    assert "does not expose the laboratory results or conclusions" in expanded
    assert "prosecution summary is an allegation" in expanded
    assert len(expanded) < 3000
