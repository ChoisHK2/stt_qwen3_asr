"""Diarization semaphore limits concurrent diarization tasks."""

import asyncio
import threading
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.session_service import SessionService


@pytest.fixture(autouse=True)
def _reset_semaphore():
    """Reset class-level semaphore between tests."""
    SessionService._diar_semaphore = None
    yield
    SessionService._diar_semaphore = None


def _make_service(max_concurrent_diar: int = 2) -> SessionService:
    """Create a SessionService with mocked dependencies."""
    store = AsyncMock()
    store.get_session_meta = AsyncMock(return_value={"sample_rate": 16000})
    store.set_status = AsyncMock()
    store.get_diar_epochs = AsyncMock(return_value=[])
    store.get_pcm = AsyncMock(return_value=b"\x00" * 32000)  # 1 sec of audio

    with patch("core.session_service.get_settings") as mock_settings:
        settings = MagicMock()
        settings.max_concurrent_diar = max_concurrent_diar
        settings.max_concurrent_asr = 32
        settings.pyannote_local_path = "/models/pyannote"
        settings.pyannote_token = None
        settings.pyannote_model = "pyannote/speaker-diarization-community-1"
        settings.diar_device = "cpu"
        settings.diar_chunk_interval_sec = 600
        settings.diar_embedding_threshold = 0.65
        mock_settings.return_value = settings
        svc = SessionService(store)

    svc.settings = settings
    return svc


@pytest.mark.asyncio
async def test_diar_semaphore_limits_concurrency():
    """At most max_concurrent_diar tasks run diarization simultaneously."""
    max_concurrent = 2
    svc = _make_service(max_concurrent_diar=max_concurrent)

    counter_lock = threading.Lock()
    counter = 0
    peak = 0

    original_diarize_sync = svc._diarize_sync

    def slow_diarize_sync(source, token, wav_path):
        nonlocal counter, peak
        with counter_lock:
            counter += 1
            if counter > peak:
                peak = counter
        time.sleep(0.1)
        with counter_lock:
            counter -= 1
        return []

    svc._diarize_sync = slow_diarize_sync

    # Use _run_diarization_remaining which uses the semaphore
    # It falls back to full diarization when no epochs exist
    with patch("core.session_service.os.path.isdir", return_value=True):
        tasks = [
            asyncio.create_task(
                svc._run_diarization_remaining(f"sess-{i}", b"\x00" * 32000, 16000)
            )
            for i in range(5)
        ]
        await asyncio.gather(*tasks)

    assert peak <= max_concurrent, (
        f"Peak concurrency {peak} exceeded limit {max_concurrent}"
    )
    # Verify all 5 tasks completed
    done_calls = [
        c for c in svc.store.set_status.call_args_list
        if c.args[1].get("diar_status") == "done"
    ]
    assert len(done_calls) == 5
