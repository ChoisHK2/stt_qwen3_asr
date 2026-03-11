from __future__ import annotations

import logging
from collections import defaultdict
from difflib import SequenceMatcher

from core.config import get_settings
from core.models import ASRSegment, DiarTurn, TimelineTurn

logger = logging.getLogger("qwen3-asr.matching")


def _overlap(a0: float, a1: float, b0: float, b1: float) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


def align_final_with_chunks(
    final_segments: list[ASRSegment],
    chunk_segments: list[ASRSegment],
) -> list[ASRSegment]:
    """STT final 텍스트(고품질)를 chunk 시간 경계(고정밀)에 맞춰 재분할한다.

    1. 각 final 세그먼트(60초)에 겹치는 chunk들을 찾는다.
    2. SequenceMatcher로 final 단어 ↔ chunk 단어를 정렬한다.
    3. 매칭된 final 단어에 chunk의 시간 경계를 상속한다.
    4. 결과: chunk 수준의 세밀한 시간 경계 + final 수준의 텍스트 품질
    """
    if not final_segments:
        return []
    if not chunk_segments:
        return final_segments

    chunks_sorted = sorted(chunk_segments, key=lambda c: c.start)
    result: list[ASRSegment] = []

    for final_seg in sorted(final_segments, key=lambda s: s.start):
        final_words = final_seg.text.split()
        if not final_words:
            continue

        # 이 final 세그먼트에 겹치는 chunk들을 찾는다
        overlapping: list[ASRSegment] = []
        for ch in chunks_sorted:
            if _overlap(final_seg.start, final_seg.end, ch.start, ch.end) > 0:
                overlapping.append(ch)

        if not overlapping:
            # chunk가 없으면 final 그대로 사용
            result.append(final_seg)
            continue

        # chunk 단어들에 시간 태그를 붙인다: (word, start_sec, end_sec)
        tagged_chunk_words: list[tuple[str, float, float]] = []
        for ch in overlapping:
            ch_words = ch.text.split()
            if not ch_words:
                continue
            ch_dur = ch.end - ch.start
            n = len(ch_words)
            for i, w in enumerate(ch_words):
                w_start = ch.start + (i / n) * ch_dur
                w_end = ch.start + ((i + 1) / n) * ch_dur
                tagged_chunk_words.append((w, w_start, w_end))

        if not tagged_chunk_words:
            result.append(final_seg)
            continue

        chunk_words_only = [t[0] for t in tagged_chunk_words]

        # SequenceMatcher로 final ↔ chunk 단어 정렬
        matcher = SequenceMatcher(None, final_words, chunk_words_only, autojunk=False)

        # final 각 단어의 추정 시간을 계산
        # 매칭된 단어 → chunk 시간 상속, 매칭되지 않은 단어 → 보간
        word_times: list[tuple[float, float]] = [(-1.0, -1.0)] * len(final_words)

        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == "equal":
                # 정확히 매칭된 단어: chunk 시간 상속
                for fi, ci in zip(range(i1, i2), range(j1, j2)):
                    word_times[fi] = (tagged_chunk_words[ci][1], tagged_chunk_words[ci][2])
            elif tag == "replace":
                # 다른 단어로 대체됨: 대응하는 chunk 시간 범위에 균등 분배
                c_start = tagged_chunk_words[j1][1]
                c_end = tagged_chunk_words[j2 - 1][2]
                n_final = i2 - i1
                dur = c_end - c_start
                for k, fi in enumerate(range(i1, i2)):
                    word_times[fi] = (
                        c_start + (k / n_final) * dur,
                        c_start + ((k + 1) / n_final) * dur,
                    )

        # "insert" (final에만 있는 단어) → 주변 매칭 단어로부터 보간
        _interpolate_missing(word_times, final_seg.start, final_seg.end)

        # chunk 경계에 맞춰 final 단어를 그룹핑
        aligned = _group_words_by_chunks(
            final_words, word_times, overlapping,
        )
        result.extend(aligned)

    logger.info(
        "Aligned %d final segments with %d chunks → %d refined segments",
        len(final_segments), len(chunk_segments), len(result),
    )
    return result


def _interpolate_missing(
    word_times: list[tuple[float, float]],
    seg_start: float,
    seg_end: float,
) -> None:
    """시간이 할당되지 않은 (-1) 단어들을 주변 값으로 보간한다 (in-place)."""
    n = len(word_times)

    for i in range(n):
        if word_times[i][0] >= 0:
            continue
        # 이전에 할당된 시간 찾기
        prev_end = seg_start
        for p in range(i - 1, -1, -1):
            if word_times[p][0] >= 0:
                prev_end = word_times[p][1]
                break
        # 다음에 할당된 시간 찾기
        next_start = seg_end
        next_idx = n
        for nx in range(i + 1, n):
            if word_times[nx][0] >= 0:
                next_start = word_times[nx][0]
                next_idx = nx
                break
        # 연속 미할당 단어 수
        gap_count = 0
        for g in range(i, min(next_idx, n)):
            if word_times[g][0] < 0:
                gap_count += 1
            else:
                break
        # 균등 분배
        dur = next_start - prev_end
        for k in range(gap_count):
            ws = prev_end + (k / gap_count) * dur
            we = prev_end + ((k + 1) / gap_count) * dur
            word_times[i + k] = (ws, we)


def _group_words_by_chunks(
    words: list[str],
    word_times: list[tuple[float, float]],
    chunks: list[ASRSegment],
) -> list[ASRSegment]:
    """시간 태그가 붙은 final 단어들을 chunk 경계에 맞춰 그룹핑한다."""
    if not words:
        return []

    # chunk 경계 생성 (start, end 쌍)
    boundaries: list[tuple[float, float]] = []
    for ch in sorted(chunks, key=lambda c: c.start):
        if boundaries and abs(ch.start - boundaries[-1][1]) < 0.01:
            # 인접 chunk → 병합하지 않고 개별 유지
            pass
        boundaries.append((ch.start, ch.end))

    if not boundaries:
        start = word_times[0][0]
        end = word_times[-1][1]
        return [ASRSegment(start=start, end=end, text=" ".join(words))]

    result: list[ASRSegment] = []
    for b_start, b_end in boundaries:
        bucket_words: list[str] = []
        for w, (ws, we) in zip(words, word_times):
            w_mid = (ws + we) / 2
            if b_start <= w_mid < b_end:
                bucket_words.append(w)
        if bucket_words:
            result.append(ASRSegment(
                start=b_start,
                end=b_end,
                text=" ".join(bucket_words),
            ))

    # 어떤 chunk 경계에도 들어가지 못한 단어 처리
    assigned = set()
    for seg in result:
        for w in seg.text.split():
            assigned.add(id(w))  # 이미 bucket에서 처리됨

    # 빈 chunk가 없고 모든 단어가 할당되었는지 확인
    all_assigned_words = sum(len(s.text.split()) for s in result)
    if all_assigned_words < len(words):
        # 미할당 단어를 마지막 세그먼트에 추가
        remaining = []
        used_count = 0
        for i, w in enumerate(words):
            w_mid = (word_times[i][0] + word_times[i][1]) / 2
            in_any = any(b_start <= w_mid < b_end for b_start, b_end in boundaries)
            if not in_any:
                remaining.append(w)
        if remaining and result:
            result[-1].text = f"{result[-1].text} {' '.join(remaining)}"
        elif remaining:
            result.append(ASRSegment(
                start=word_times[0][0],
                end=word_times[-1][1],
                text=" ".join(remaining),
            ))

    return result


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
