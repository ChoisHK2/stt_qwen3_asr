from __future__ import annotations

from collections import defaultdict

from core.config import get_settings
from core.models import ASRSegment, DiarTurn, TimelineTurn


def _overlap(a0: float, a1: float, b0: float, b1: float) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


def _distribute_words_by_time(
    words: list[str],
    seg_start: float,
    seg_end: float,
    query_start: float,
    query_end: float,
) -> list[str]:
    """ASR 세그먼트 내의 단어들을 시간 비율에 따라 분배한다.

    word-level timestamp가 없으므로 세그먼트 내에서 균등 분포를 가정하고,
    단어의 중심점(midpoint)이 query 범위 안에 있으면 포함한다.
    이렇게 하면 각 단어가 정확히 하나의 diar turn에만 할당된다.
    """
    if not words:
        return []
    seg_dur = seg_end - seg_start
    if seg_dur <= 0:
        return words

    n = len(words)
    result = []
    for i, w in enumerate(words):
        # 각 단어의 추정 중심점 (균등 분포)
        w_mid = seg_start + ((i + 0.5) / n) * seg_dur
        # 중심점이 query 범위 안에 있으면 포함
        if query_start <= w_mid < query_end:
            result.append(w)
    return result


def map_speakers(segments: list[ASRSegment], diar: list[DiarTurn]) -> list[TimelineTurn]:
    """Diarization 결과를 기준으로 ASR 텍스트를 화자별로 분배한다.

    기존 방식: ASR 세그먼트별로 가장 겹침이 큰 화자 1명 배정
    새 방식: diar turn별로 해당 시간 구간의 ASR 텍스트를 추출하여 배정

    diar가 없으면 ASR 세그먼트 기준 fallback.
    """
    settings = get_settings()

    if not diar:
        # diarization이 없으면 기존 방식 fallback
        turns = [
            TimelineTurn(speaker="UNKNOWN", start=seg.start, end=seg.end, text=seg.text)
            for seg in segments
        ]
        return merge_turns(turns)

    if not segments:
        return []

    # diar turn 기준으로 텍스트 분배
    turns: list[TimelineTurn] = []
    for d in sorted(diar, key=lambda x: x.start):
        collected_words: list[str] = []

        for seg in segments:
            ov = _overlap(seg.start, seg.end, d.start, d.end)
            if ov <= 0:
                continue

            seg_words = seg.text.split()
            if not seg_words:
                continue

            # ASR 세그먼트에서 diar turn 시간에 해당하는 단어를 추출
            extracted = _distribute_words_by_time(
                seg_words, seg.start, seg.end, d.start, d.end,
            )
            collected_words.extend(extracted)

        text = " ".join(collected_words).strip()
        if text:
            turns.append(TimelineTurn(
                speaker=d.speaker,
                start=d.start,
                end=d.end,
                text=text,
            ))

    # merge 먼저 수행 후, 긴 turn을 분할 (분할 결과가 다시 병합되지 않도록)
    turns = merge_turns(turns)
    turns = _split_long_turns(turns, settings.max_turn_sec)

    return turns


def _split_long_turns(turns: list[TimelineTurn], max_sec: float) -> list[TimelineTurn]:
    """max_sec(기본 15초)를 초과하는 turn을 분할한다.

    같은 화자의 연속 발화가 길 때, max_sec 단위로 잘라서 표시한다.
    """
    if max_sec <= 0:
        return turns

    result: list[TimelineTurn] = []
    for turn in turns:
        dur = turn.end - turn.start
        if dur <= max_sec:
            result.append(turn)
            continue

        words = turn.text.split()
        if not words:
            result.append(turn)
            continue

        # 시간 비율로 분할
        n_splits = max(1, int(dur / max_sec) + (1 if dur % max_sec > 0 else 0))
        words_per_split = max(1, len(words) // n_splits)

        for i in range(n_splits):
            split_start = turn.start + (i / n_splits) * dur
            split_end = turn.start + ((i + 1) / n_splits) * dur
            w_start = i * words_per_split
            w_end = (i + 1) * words_per_split if i < n_splits - 1 else len(words)
            split_words = words[w_start:w_end]
            if split_words:
                result.append(TimelineTurn(
                    speaker=turn.speaker,
                    start=round(split_start, 3),
                    end=round(split_end, 3),
                    text=" ".join(split_words),
                ))

    return result


def merge_turns(turns: list[TimelineTurn]) -> list[TimelineTurn]:
    s = get_settings()
    if s.merge_mode == "none":
        return turns
    merged: list[TimelineTurn] = []
    for turn in turns:
        if not merged:
            merged.append(turn)
            continue
        prev = merged[-1]
        gap = turn.start - prev.end
        can_merge = turn.speaker == prev.speaker and gap <= s.merge_gap_sec
        if can_merge and s.merge_mode in {"gap", "duration", "sentence"}:
            prev.end = turn.end
            prev.text = f"{prev.text} {turn.text}".strip()
        else:
            merged.append(turn)
    return [t for t in merged if (t.end - t.start) >= s.min_turn_sec or len(t.text.split()) >= s.min_words_per_turn]
