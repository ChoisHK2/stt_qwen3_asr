"""Diarization semaphore limits concurrent diarization tasks."""

import asyncio
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

    with patch("core.session_service.get_settings") as mock_settings:
        settings = MagicMock()
        settings.max_concurrent_diar = max_concurrent_diar
        settings.pyannote_local_path = "/models/pyannote"
        settings.pyannote_token = None
        settings.pyannote_model = "pyannote/speaker-diarization-community-1"
        settings.diar_device = "cpu"
        mock_settings.return_value = settings
        svc = SessionService(store)

    # Restore real settings for attribute access
    svc.settings = settings
    return svc


@pytest.mark.asyncio
async def test_diar_semaphore_limits_concurrency():
    """At most max_concurrent_diar tasks run diarization simultaneously."""
    max_concurrent = 2
    svc = _make_service(max_concurrent_diar=max_concurrent)

    peak_concurrent = 0
    current_concurrent = 0
    lock = asyncio.Lock()

    original_diarize_sync = svc._diarize_sync

    def slow_diarize_sync(source, token, wav_path):
        nonlocal peak_concurrent, current_concurrent
        # We can't use asyncio.Lock in sync code, use a simple counter
        # This is approximate but sufficient for testing
        import threading
        nonlocal lock
        current_concurrent_local = 0
        # Use threading lock for thread-safe counter
        if not hasattr(slow_diarize_sync, '_lock'):
            slow_diarize_sync._lock = asyncio.Lock()
            slow_diarize_sync._counter = 0
            slow_diarize_sync._peak = 0
            slow_diarize_sync._tlock = threading.Lock()

        with slow_diarize_sync._tlock:
            slow_diarize_sync._counter += 1
            if slow_diarize_sync._counter > slow_diarize_sync._peak:
                slow_diarize_sync._peak = slow_diarize_sync._counter

        import time
        time.sleep(0.1)  # Simulate work

        with slow_diarize_sync._tlock:
            slow_diarize_sync._counter -= 1

        return []

    svc._diarize_sync = slow_diarize_sync

    # Mock os.path.isdir to return True for pyannote path
    with patch("core.session_service.os.path.isdir", return_value=True):
        tasks = [
            asyncio.create_task(svc._run_diarization_background(f"sess-{i}", f"/tmp/test-{i}.wav"))
            for i in range(5)
        ]
        await asyncio.gather(*tasks)

    assert slow_diarize_sync._peak <= max_concurrent, (
        f"Peak concurrency {slow_diarize_sync._peak} exceeded limit {max_concurrent}"
    )
    # Verify all 5 tasks completed (set_status called with "done" for each)
    done_calls = [
        c for c in svc.store.set_status.call_args_list
        if c.args[1].get("diar_status") == "done"
    ]
    assert len(done_calls) == 5
