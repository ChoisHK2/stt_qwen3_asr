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

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def post(self, url, data=None, files=None):
        self.capture["url"] = url
        self.capture["data"] = data
        self.capture["files"] = files
        return DummyResponse(self.payload)


def test_transcribe_partial_uses_multipart_and_parses_segments(monkeypatch):
    capture = {}

    def fake_client(*args, **kwargs):
        return DummyClient({"segments": [{"start": 0.0, "end": 1.0, "text": "hello"}]}, capture)

    monkeypatch.setattr("clients.asr_client.httpx.AsyncClient", fake_client)
    cli = ASRClient()
    audio = np.zeros(16000, dtype=np.float32)

    segs, err = asyncio.run(cli.transcribe_partial(audio, 16000))

    assert err is None
    assert len(segs) == 1
    assert segs[0].text == "hello"
    assert capture["data"]["response_format"] == "verbose_json"
    assert "file" in capture["files"]
