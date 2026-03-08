import asyncio

import numpy as np

from clients.asr_client import ASRClient


class DummyResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


class DummyClient:
    def __init__(self, payload, capture):
        self.payload = payload
        self.capture = capture
        self.is_closed = False

    async def post(self, url, json=None, **kwargs):
        self.capture["url"] = url
        self.capture["json"] = json
        return DummyResponse(self.payload)

    async def aclose(self):
        self.is_closed = True


def test_transcribe_partial_uses_chat_completions_and_parses_response():
    capture = {}
    dummy_client = DummyClient(
        {"choices": [{"message": {"content": "hello"}}]},
        capture,
    )

    cli = ASRClient()
    cli._client = dummy_client
    audio = np.zeros(16000, dtype=np.float32)

    segs, err = asyncio.run(cli.transcribe_partial(audio, 16000))

    assert err is None
    assert len(segs) == 1
    assert segs[0].text == "hello"
    assert "/v1/chat/completions" in capture["url"]
    messages = capture["json"]["messages"]
    assert messages[0]["role"] == "user"
    content = messages[0]["content"]
    assert content[0]["type"] == "audio_url"
    assert content[0]["audio_url"]["url"].startswith("data:audio/wav;base64,")
