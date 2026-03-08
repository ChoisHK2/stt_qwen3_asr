import asyncio

import numpy as np

from clients.asr_client import ASRClient, parse_asr_output


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

    async def post(self, url, json=None, headers=None, **kwargs):
        self.capture["url"] = url
        self.capture["json"] = json
        self.capture["headers"] = headers
        return DummyResponse(self.payload)

    async def aclose(self):
        self.is_closed = True


def test_transcribe_partial_uses_chat_completions_and_parses_response():
    capture = {}
    dummy_client = DummyClient(
        {
            "choices": [
                {
                    "message": {
                        "content": "<|en|><|transcription|>hello<|endoftext|>",
                    }
                }
            ]
        },
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
    # payload should contain model and messages with audio_url
    payload = capture["json"]
    assert "model" in payload
    assert payload["messages"][0]["content"][0]["type"] == "audio_url"
    assert payload["messages"][0]["content"][0]["audio_url"]["url"].startswith("data:audio/wav;base64,")


def test_parse_asr_output_with_tags():
    lang, text = parse_asr_output("<|ko|><|transcription|>안녕하세요<|endoftext|>")
    assert lang == "ko"
    assert text == "안녕하세요"


def test_parse_asr_output_plain_text():
    lang, text = parse_asr_output("hello world")
    assert lang is None
    assert text == "hello world"


def test_parse_asr_output_with_language_only():
    lang, text = parse_asr_output("<|en|>hello world")
    assert lang == "en"
    assert text == "hello world"
