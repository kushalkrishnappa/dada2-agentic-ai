from rag import config


def test_chat_has_a_request_timeout(monkeypatch):
    # With no timeout one stalled request to Gemini spins the app forever.
    monkeypatch.setattr(config, "CHAT_BACKEND", "gemini")
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
    model = config.chat()
    assert model.timeout == config.CHAT_TIMEOUT == 60
    assert model.max_retries == 6       # the library default, left alone
