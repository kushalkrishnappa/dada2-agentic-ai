"""A stand-in for config.chat(), so the module logic is tested without
spending quota. replies maps a Pydantic class to the instance the fake
model should return for with_structured_output(that class)."""
import types

import pytest


class FakeChat:
    def __init__(self, replies=None, text="an answer [A.txt]"):
        self.replies = replies or {}
        self.text = text
        self.prompts = []

    def with_structured_output(self, schema):
        chat = self

        class Bound:
            def invoke(self, prompt):
                chat.prompts.append(prompt)
                reply = chat.replies[schema]
                return reply.pop(0) if isinstance(reply, list) else reply
        return Bound()

    def invoke(self, prompt):
        self.prompts.append(prompt)
        return types.SimpleNamespace(content=self.text)


@pytest.fixture
def fake_chat(monkeypatch):
    from rag import config

    def install(replies=None, text="an answer [A.txt]"):
        chat = FakeChat(replies, text)
        monkeypatch.setattr(config, "chat", lambda: chat)
        return chat
    return install
