from bencheck.integrations.llm.base import LLMRequest
from bencheck.integrations.llm.factory import create_provider
from bencheck.integrations.llm.settings import LLMSettings


def test_gemini_stub_response_contains_prompt_fragment():
    provider = create_provider("gemini", api_key="stub-key", model="gemini-demo")
    request = LLMRequest(prompt="Hello world", temperature=0.2, max_tokens=32)
    response = provider.complete(request)

    assert response.text.startswith("[gemini-demo]")
    assert "Hello world" in response.text
    assert response.raw["provider"] == "gemini"


def test_llm_settings_read_from_environment(monkeypatch):
    monkeypatch.setenv("BENCHECK_LLM_PROVIDER", "gemini")
    monkeypatch.setenv("BENCHECK_LLM_MODEL", "gemini-1.5-flash")
    monkeypatch.setenv("GOOGLE_API_KEY", "abc123")

    settings = LLMSettings.from_env()

    assert settings.provider == "gemini"
    assert settings.model == "gemini-1.5-flash"
    assert settings.api_key == "abc123"

    monkeypatch.delenv("BENCHECK_LLM_PROVIDER")
    monkeypatch.delenv("BENCHECK_LLM_MODEL")
    monkeypatch.delenv("GOOGLE_API_KEY")
