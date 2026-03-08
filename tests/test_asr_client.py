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

    async def post(self, url, files=None, data=None, **kwargs):
        self.capture["url"] = url
        self.capture["files"] = files
        self.capture["data"] = data
        return DummyResponse(self.payload)

    async def aclose(self):
        self.is_closed = True


def test_transcribe_partial_uses_transcriptions_endpoint_and_parses_response():
    capture = {}
    dummy_client = DummyClient(
        {"text": "hello"},
        capture,
    )

    cli = ASRClient()
    cli._client = dummy_client
    audio = np.zeros(16000, dtype=np.float32)

    segs, err = asyncio.run(cli.transcribe_partial(audio, 16000))

    assert err is None
    assert len(segs) == 1
    assert segs[0].text == "hello"
    assert "/v1/audio/transcriptions" in capture["url"]
    # multipart: file field should be a tuple (filename, bytes, content_type)
    assert "file" in capture["files"]
    file_tuple = capture["files"]["file"]
    assert file_tuple[0] == "audio.wav"
    assert file_tuple[2] == "audio/wav"
    # model should be sent as form data
    assert "model" in capture["data"]
