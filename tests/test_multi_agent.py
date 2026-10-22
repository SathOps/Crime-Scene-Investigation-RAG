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
