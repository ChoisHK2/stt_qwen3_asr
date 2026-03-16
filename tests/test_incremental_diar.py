"""Tests for incremental diarization and embedding-based speaker matching."""

import numpy as np
import pytest

from clients.diarization_client import (
    DiarEpochResult,
    match_speakers_by_embedding,
)
from core.matching import (
    TimelineTurn,
    _final_merge_same_speaker,
    _split_long_turns,
    map_speakers,
)
from core.models import ASRSegment, DiarTurn


# ── Embedding-based speaker matching ────────────────────────────


def test_match_speakers_by_embedding_exact_match():
    """동일한 임베딩은 같은 화자로 매핑되어야 한다."""
    emb_a = np.array([1.0, 0.0, 0.0])
    emb_b = np.array([0.0, 1.0, 0.0])

    prev = {"SPEAKER_00": emb_a, "SPEAKER_01": emb_b}
    curr = {"SPEAKER_0": emb_a, "SPEAKER_1": emb_b}

    mapping, updated = match_speakers_by_embedding(prev, curr, threshold=0.5)
    assert mapping["SPEAKER_0"] == "SPEAKER_00"
    assert mapping["SPEAKER_1"] == "SPEAKER_01"
    # EMA 업데이트된 임베딩이 반환되어야 한다
    assert "SPEAKER_00" in updated
    assert "SPEAKER_01" in updated


def test_match_speakers_by_embedding_no_match():
    """유사도가 threshold 미만이면 매핑되지 않아야 한다."""
    prev = {"SPEAKER_00": np.array([1.0, 0.0, 0.0])}
    curr = {"SPEAKER_0": np.array([0.0, 1.0, 0.0])}  # orthogonal

    mapping, updated = match_speakers_by_embedding(prev, curr, threshold=0.5)
    assert len(mapping) == 0
    # 매칭 실패 시 기존 임베딩 유지
    assert "SPEAKER_00" in updated


def test_match_speakers_by_embedding_partial_match():
    """일부만 매칭되고 나머지는 새 화자여야 한다."""
    prev = {"SPEAKER_00": np.array([1.0, 0.0, 0.0])}
    curr = {
        "SPEAKER_0": np.array([0.98, 0.1, 0.0]),  # close to SPEAKER_00
        "SPEAKER_1": np.array([0.0, 0.0, 1.0]),    # new speaker
    }

    mapping, updated = match_speakers_by_embedding(prev, curr, threshold=0.5)
    assert mapping.get("SPEAKER_0") == "SPEAKER_00"
    assert "SPEAKER_1" not in mapping


def test_match_speakers_empty_prev():
    """이전 에폭이 비어있으면 빈 매핑을 반환해야 한다."""
    curr = {"SPEAKER_0": np.array([1.0, 0.0])}
    mapping, updated = match_speakers_by_embedding({}, curr, threshold=0.5)
    assert mapping == {}
    assert updated == {}


def test_match_speakers_empty_curr():
    """현재 에폭이 비어있으면 빈 매핑을 반환해야 한다."""
    prev = {"SPEAKER_00": np.array([1.0, 0.0])}
    mapping, updated = match_speakers_by_embedding(prev, {}, threshold=0.5)
    assert mapping == {}
    assert "SPEAKER_00" in updated


def test_match_speakers_ema_blending():
    """EMA 누적으로 임베딩이 점진적으로 업데이트되어야 한다."""
    prev = {"SPEAKER_00": np.array([1.0, 0.0, 0.0])}
    curr = {"SPEAKER_0": np.array([0.9, 0.1, 0.0])}

    mapping, updated = match_speakers_by_embedding(prev, curr, threshold=0.5, ema_alpha=0.3)
    assert "SPEAKER_00" in mapping.values()
    # EMA: (0.7 * [1,0,0] + 0.3 * [0.9,0.1,0]) = [0.97, 0.03, 0]
    # L2 정규화 후에도 원래 방향에 가까워야 함
    blended = updated["SPEAKER_00"]
    assert blended[0] > blended[1]  # 여전히 첫 번째 차원이 지배적


# ── Final merge same speaker ────────────────────────────────────


def test_final_merge_same_speaker():
    """연속 동일 화자 turn이 하나로 병합되어야 한다."""
    turns = [
        TimelineTurn(speaker="SPEAKER_00", start=0.0, end=5.0, text="hello"),
        TimelineTurn(speaker="SPEAKER_00", start=5.0, end=10.0, text="world"),
        TimelineTurn(speaker="SPEAKER_01", start=10.0, end=15.0, text="hi"),
        TimelineTurn(speaker="SPEAKER_00", start=15.0, end=20.0, text="again"),
    ]
    merged = _final_merge_same_speaker(turns)
    assert len(merged) == 3
    assert merged[0].speaker == "SPEAKER_00"
    assert merged[0].text == "hello world"
    assert merged[0].end == 10.0
    assert merged[1].speaker == "SPEAKER_01"
    assert merged[2].speaker == "SPEAKER_00"
    assert merged[2].text == "again"


def test_final_merge_all_same_speaker():
    """모든 turn이 같은 화자면 하나로 병합되어야 한다."""
    turns = [
        TimelineTurn(speaker="SPEAKER_00", start=0.0, end=5.0, text="a"),
        TimelineTurn(speaker="SPEAKER_00", start=5.0, end=10.0, text="b"),
        TimelineTurn(speaker="SPEAKER_00", start=10.0, end=15.0, text="c"),
    ]
    merged = _final_merge_same_speaker(turns)
    assert len(merged) == 1
    assert merged[0].text == "a b c"
    assert merged[0].end == 15.0


def test_final_merge_empty():
    assert _final_merge_same_speaker([]) == []


def test_split_then_merge_preserves_single_speaker():
    """split_long_turns 후에도 동일 화자는 _final_merge_same_speaker로 하나가 되어야 한다."""
    turns = [
        TimelineTurn(speaker="SPEAKER_00", start=0.0, end=30.0,
                     text=" ".join(f"word{i}" for i in range(30))),
    ]
    split = _split_long_turns(turns, max_sec=15.0)
    assert len(split) == 2  # 30초 → 15초씩 2개로 분할

    # 재병합
    merged = _final_merge_same_speaker(split)
    assert len(merged) == 1
    assert merged[0].speaker == "SPEAKER_00"


# ── map_speakers with merge ─────────────────────────────────────


def test_map_speakers_merges_consecutive_same_speaker():
    """map_speakers 결과에서 연속 동일 화자가 병합되어야 한다."""
    segments = [
        ASRSegment(start=0.0, end=5.0, text="hello world"),
        ASRSegment(start=5.0, end=10.0, text="how are you"),
        ASRSegment(start=10.0, end=15.0, text="doing today"),
    ]
    # 모두 같은 화자
    diar = [
        DiarTurn(speaker="SPEAKER_00", start=0.0, end=15.0),
    ]
    result = map_speakers(segments, diar)

    # 동일 화자이므로 1개로 병합되어야 함
    assert len(result) == 1
    assert result[0].speaker == "SPEAKER_00"
    assert "hello" in result[0].text


# ── Config tests ────────────────────────────────────────────────


def test_config_diar_chunk_interval():
    from core.config import Settings
    s = Settings()
    assert s.diar_chunk_interval_sec == 600
    assert s.diar_embedding_threshold == 0.45


def test_config_diar_chunk_interval_custom():
    from core.config import Settings
    s = Settings(diar_chunk_interval_sec=300, diar_embedding_threshold=0.7)
    assert s.diar_chunk_interval_sec == 300
    assert s.diar_embedding_threshold == 0.7
