"""Regression tests for workflow telemetry and serialized state."""

from models.schemas import WorkflowState


def test_workflow_metrics_are_serialized():
    state = WorkflowState(query="Summarize the case")
    state.metrics.update({
        "retrieval_duration_seconds": 0.12,
        "specialist_agents_selected": 2,
        "specialist_agents_completed": 1,
        "source_count": 3,
    })

    payload = state.to_dict()

    assert payload["metrics"]["retrieval_duration_seconds"] == 0.12
    assert payload["metrics"]["specialist_agents_selected"] == 2
    assert payload["metrics"]["specialist_agents_completed"] == 1
    assert payload["metrics"]["source_count"] == 3


def test_workflow_metrics_default_to_empty_mapping():
    state = WorkflowState()

    assert state.metrics == {}
    assert state.to_dict()["metrics"] == {}
