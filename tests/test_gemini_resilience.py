import pytest
from unittest.mock import MagicMock
from google.genai.types import Model

from llm.llm_service import LLMService

def test_resolve_gemini_model():
    assert LLMService.resolve_gemini_model("models/gemini-2.0-flash") == "gemini-2.0-flash"
    assert LLMService.resolve_gemini_model("gemini-2.5-flash") == "gemini-2.0-flash"
    assert LLMService.resolve_gemini_model("gemini-3.8-flash") == "gemini-2.0-flash"
    assert LLMService.resolve_gemini_model("gemini-1.5-flash") == "gemini-1.5-flash"
    assert LLMService.resolve_gemini_model(None) == "gemini-2.0-flash"

def test_gemini_dynamic_discovery_and_cascade():
    mock_client = MagicMock()
    mock_client.models.list.return_value = [
        Model(name="models/gemini-2.0-flash", supported_actions=["generateContent"]),
        Model(name="models/gemini-1.5-flash-latest", supported_actions=["generateContent"]),
    ]

    attempted_models = []
    def mock_generate(model, contents):
        attempted_models.append(model)
        if model == "gemini-1.5-flash":
            raise Exception("404 NOT_FOUND: models/gemini-1.5-flash is not found for API version v1beta")
        resp = MagicMock()
        resp.text = f"Validated response from {model}"
        return resp

    mock_client.models.generate_content.side_effect = mock_generate

    res, used_model = LLMService._call_gemini_with_fallback_and_retry(
        client=mock_client,
        primary_model="gemini-1.5-flash",
        contents="Research query"
    )

    assert used_model in ["gemini-1.5-flash-latest", "gemini-2.0-flash"]
    assert "gemini-1.5-flash" in attempted_models
    assert res.text == f"Validated response from {used_model}"

def test_gemini_multi_error_cascade():
    """Verify that 503 on gemini-2.0-flash and 404 on gemini-1.5-flash seamlessly cascade to gemini-2.0-flash-exp."""
    mock_client = MagicMock()
    mock_client.models.list.return_value = []

    attempted = []
    def mock_generate(model, contents):
        attempted.append(model)
        if model == "gemini-2.0-flash":
            raise Exception("503 UNAVAILABLE: Model is experiencing high demand")
        if model == "gemini-1.5-flash":
            raise Exception("404 NOT_FOUND: models/gemini-1.5-flash is not found for API version v1beta")
        if model == "gemini-2.0-flash-exp":
            resp = MagicMock()
            resp.text = "Success from exp model"
            return resp
        raise Exception("404 not found")

    mock_client.models.generate_content.side_effect = mock_generate

    res, used_model = LLMService._call_gemini_with_fallback_and_retry(
        client=mock_client,
        primary_model="gemini-2.0-flash",
        contents="Analyze evidence"
    )

    assert used_model == "gemini-2.0-flash-exp"
    assert res.text == "Success from exp model"
    assert "gemini-2.0-flash" in attempted
    assert "gemini-2.0-flash-exp" in attempted

def test_gemini_filters_non_text_models_and_handles_modality_mismatch():
    """Verify that TTS, audio, and imagen models are filtered out from candidates and modality errors failover."""
    mock_client = MagicMock()
    mock_client.models.list.return_value = [
        Model(name="models/gemini-2.5-flash-preview-tts", supported_actions=["generateContent"]),
        Model(name="models/imagen-3.0-generate-002", supported_actions=["generateContent"]),
        Model(name="models/gemini-2.0-flash", supported_actions=["generateContent"]),
    ]

    attempted = []
    def mock_generate(model, contents):
        attempted.append(model)
        if "tts" in model:
            raise Exception("400: The requested combination of response modalities (TEXT) is not supported by the model.")
        resp = MagicMock()
        resp.text = "Success from text model"
        return resp

    mock_client.models.generate_content.side_effect = mock_generate

    res, used_model = LLMService._call_gemini_with_fallback_and_retry(
        client=mock_client,
        primary_model="gemini-2.0-flash",
        contents="Analyze evidence"
    )

    assert "gemini-2.5-flash-preview-tts" not in attempted
    assert used_model == "gemini-2.0-flash"
    assert res.text == "Success from text model"

