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


def _merge_same_speaker_diar(diar: list[DiarTurn]) -> list[DiarTurn]:
    """연속 동일 화자의 diar turn을 하나로 병합한다.

    pyannote가 같은 화자를 여러 짧은 turn으로 쪼개는 경우가 많다.
    매칭 전에 병합하면 텍스트 분배가 정확해지고, 최종 결과도 깔끔해진다.
    """
    if not diar:
        return []
    sorted_diar = sorted(diar, key=lambda d: d.start)
    merged: list[DiarTurn] = [DiarTurn(
        speaker=sorted_diar[0].speaker,
        start=sorted_diar[0].start,
        end=sorted_diar[0].end,
    )]
    for d in sorted_diar[1:]:
        prev = merged[-1]
        if d.speaker == prev.speaker:
            # 동일 화자 → 시간 확장 (사이 갭 포함)
            prev.end = max(prev.end, d.end)
        else:
            merged.append(DiarTurn(speaker=d.speaker, start=d.start, end=d.end))
    return merged


def _resolve_overlapping_turns(diar: list[DiarTurn], short_threshold: float = 2.0) -> list[DiarTurn]:
    """겹치는 diar turn을 해소한다.

    pyannote는 동시 발화(overlapping speech)를 감지하여 같은 시간대에
    여러 화자의 turn을 반환할 수 있다. 이를 그대로 사용하면 같은 단어가
    여러 화자에게 중복 분배된다.

    전략:
    - 2초 이하의 짧은 발화가 다른 화자와 겹치면, 앞뒤 문맥(이전/다음 turn의
      화자)을 보고 dominant 화자에게 흡수시킨다.
    - 2초 초과의 긴 겹침은 시간 중간점으로 분할하여 각 화자에게 배분한다.
    """
    if len(diar) <= 1:
        return diar

    sorted_diar = sorted(diar, key=lambda d: d.start)

    # 겹침이 있는지 빠른 체크
    has_overlap = False
    for i in range(len(sorted_diar) - 1):
        if sorted_diar[i].end > sorted_diar[i + 1].start:
            has_overlap = True
            break
    if not has_overlap:
        return sorted_diar

    # 겹치는 turn 해소
    result: list[DiarTurn] = []
    skip: set[int] = set()

    for i, curr in enumerate(sorted_diar):
        if i in skip:
            continue

        # 현재 turn과 겹치는 다음 turn들을 찾는다
        overlaps: list[int] = []
        for j in range(i + 1, len(sorted_diar)):
            if j in skip:
                continue
            if sorted_diar[j].start < curr.end:
                overlaps.append(j)
            else:
                break  # 정렬되어 있으므로 이후는 겹치지 않음

        if not overlaps:
            result.append(DiarTurn(speaker=curr.speaker, start=curr.start, end=curr.end))
            continue

        # 겹치는 turn들과 함께 처리
        group = [curr] + [sorted_diar[j] for j in overlaps]

        # 각 turn의 길이 계산
        durations = [(t.end - t.start) for t in group]

        # 이전/다음 non-overlapping turn의 화자 확인 (문맥)
        prev_speaker = result[-1].speaker if result else None
        next_speaker = None
        for k in range(max(overlaps) + 1, len(sorted_diar)):
            if k not in skip:
                next_speaker = sorted_diar[k].speaker
                break

        # dominant 화자: 가장 긴 발화를 가진 화자
        dominant_idx = durations.index(max(durations))
        dominant = group[dominant_idx]

        # 짧은 turn(≤ threshold)이 겹치면 문맥 기반으로 흡수
        all_resolved = True
        kept_turns: list[DiarTurn] = []

        for gi, t in enumerate(group):
            dur = durations[gi]
            if gi == dominant_idx:
                kept_turns.append(DiarTurn(speaker=t.speaker, start=t.start, end=t.end))
                continue

            if dur <= short_threshold:
                # 짧은 발화: 앞뒤 문맥을 보고 dominant에 흡수할지 결정
                # 앞뒤 화자가 이 짧은 turn과 같은 화자이면 유지
                context_match = (t.speaker == prev_speaker or t.speaker == next_speaker)
                if context_match and t.speaker != dominant.speaker:
                    # 문맥상 이 화자가 실제로 있었을 가능성 → 겹침 구간만 정리
                    ov_start = max(t.start, dominant.start)
                    ov_end = min(t.end, dominant.end)
                    mid = (ov_start + ov_end) / 2

                    # 짧은 turn을 겹치지 않는 부분만 남김
                    if t.start < mid:
                        kept_turns.append(DiarTurn(speaker=t.speaker, start=t.start, end=mid))
                    if mid < t.end:
                        # dominant의 시작을 mid로 조정 (아래에서 처리)
                        pass
                else:
                    # dominant에 흡수 (짧은 발화 제거)
                    logger.debug(
                        "Absorbing short overlap: %s [%.1f-%.1f] (%.1fs) into %s",
                        t.speaker, t.start, t.end, dur, dominant.speaker,
                    )
            else:
                # 긴 겹침: 시간 중간점으로 분할
                all_resolved = False
                ov_start = max(t.start, dominant.start)
                ov_end = min(t.end, dominant.end)
                mid = (ov_start + ov_end) / 2

                # 각자 겹치지 않는 영역 유지
                if t.start < mid:
                    kept_turns.append(DiarTurn(speaker=t.speaker, start=t.start, end=mid))
                if mid < t.end:
                    kept_turns.append(DiarTurn(speaker=t.speaker, start=mid, end=t.end))

        # dominant turn의 시간 조정: 다른 kept_turns과 겹치지 않도록
        for kt in kept_turns:
            if kt.speaker == dominant.speaker:
                continue
            # dominant와 겹치는 부분 조정
            for dt in kept_turns:
                if dt.speaker != dominant.speaker:
                    continue
                if dt.start < kt.end and dt.end > kt.start:
                    # 겹침 해소: dominant의 겹치는 부분 제거
                    if dt.start < kt.start:
                        dt.end = min(dt.end, kt.start)
                    elif dt.end > kt.end:
                        dt.start = max(dt.start, kt.end)

        result.extend(sorted(kept_turns, key=lambda t: t.start))
        skip.update(overlaps)

    # 빈 turn 제거 및 정렬
    result = [t for t in result if t.end > t.start + 0.01]
    result.sort(key=lambda t: t.start)
    return result


def _fill_diar_gaps(diar: list[DiarTurn], audio_end: float) -> list[DiarTurn]:
    """diar turn 사이의 갭을 앞뒤 화자에게 할당하여 텍스트 누락을 방지한다.

    갭의 중간점을 기준으로 앞 turn과 뒤 turn에 나눠 할당한다.
    - 오디오 시작 전 갭: 첫 turn에 할당
    - 오디오 끝 후 갭: 마지막 turn에 할당
    """
    if not diar:
        return []

    filled = [DiarTurn(speaker=d.speaker, start=d.start, end=d.end) for d in diar]

    # 시작 부분 갭: 첫 turn이 0초부터 시작하지 않으면
    if filled[0].start > 0:
        filled[0].start = 0.0

    # turn 사이 갭 처리
    for i in range(len(filled) - 1):
        curr = filled[i]
        nxt = filled[i + 1]
        if nxt.start > curr.end:
            gap_mid = (curr.end + nxt.start) / 2
            curr.end = gap_mid
            nxt.start = gap_mid

    # 끝 부분 갭: 마지막 turn 이후 남은 오디오
    if audio_end > filled[-1].end:
        filled[-1].end = audio_end

    return filled


def map_speakers(segments: list[ASRSegment], diar: list[DiarTurn]) -> list[TimelineTurn]:
    """Diarization 결과를 기준으로 ASR 텍스트를 화자별로 분배한다.

    1. 연속 동일 화자 diar turn 병합
    2. diar 갭을 앞뒤 화자에 할당 (텍스트 누락 방지)
    3. 병합된 diar turn 기준으로 ASR 텍스트 분배

    diar가 없으면 ASR 세그먼트 기준 fallback.
    """
    settings = get_settings()

    if not diar:
        turns = [
            TimelineTurn(speaker="UNKNOWN", start=seg.start, end=seg.end, text=seg.text)
            for seg in segments
        ]
        return merge_turns(turns)

    if not segments:
        return []

    # 1) 겹치는 turn 해소 (pyannote의 overlapping speech 처리)
    resolved_diar = _resolve_overlapping_turns(diar)

    # 2) 연속 동일 화자 병합
    merged_diar = _merge_same_speaker_diar(resolved_diar)

    # 3) 갭 채우기 (오디오 끝 시간 = 마지막 ASR 세그먼트 끝)
    audio_end = max(seg.end for seg in segments)
    filled_diar = _fill_diar_gaps(merged_diar, audio_end)

    logger.info(
        "Diar preprocessing: %d raw → %d resolved → %d merged → %d gap-filled turns",
        len(diar), len(resolved_diar), len(merged_diar), len(filled_diar),
    )

    # 3) diar turn 기준으로 텍스트 분배
    turns: list[TimelineTurn] = []
    for d in filled_diar:
        collected_words: list[str] = []

        for seg in segments:
            ov = _overlap(seg.start, seg.end, d.start, d.end)
            if ov <= 0:
                continue

            seg_words = seg.text.split()
            if not seg_words:
                continue

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

    # merge → split → 재merge (split 후 동일 화자 연속 방지) → 최종 필터
    turns = merge_turns(turns)
    turns = _split_long_turns(turns, settings.max_turn_sec)
    turns = _final_merge_same_speaker(turns)
    turns = _filter_short_turns(turns, settings.min_turn_sec, settings.min_words_per_turn)

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


def _final_merge_same_speaker(turns: list[TimelineTurn]) -> list[TimelineTurn]:
    """최종 출력에서 연속 동일 화자 turn을 무조건 병합한다.

    _split_long_turns 이후나, diar에서 동일 화자가 연속으로 나오는 경우
    gap에 관계없이 동일 화자 연속 turn을 하나로 합쳐 표시한다.
    """
    if not turns:
        return []
    merged: list[TimelineTurn] = [TimelineTurn(
        speaker=turns[0].speaker,
        start=turns[0].start,
        end=turns[0].end,
        text=turns[0].text,
    )]
    for turn in turns[1:]:
        prev = merged[-1]
        if turn.speaker == prev.speaker:
            prev.end = turn.end
            prev.text = f"{prev.text} {turn.text}".strip()
        else:
            merged.append(TimelineTurn(
                speaker=turn.speaker,
                start=turn.start,
                end=turn.end,
                text=turn.text,
            ))
    return merged


def _filter_short_turns(
    turns: list[TimelineTurn],
    min_sec: float,
    min_words: int,
) -> list[TimelineTurn]:
    """min_turn_sec 미만이고 min_words_per_turn 미만인 턴을 인접 턴에 흡수한다.

    단순 삭제하면 텍스트가 유실되므로, 짧은 턴의 텍스트를 시간적으로
    가장 가까운 인접 턴(직전 우선)에 병합한다.
    """
    if not turns:
        return []

    keep: list[bool] = [
        (t.end - t.start) >= min_sec or len(t.text.split()) >= min_words
        for t in turns
    ]

    # 모두 통과하면 그대로 반환
    if all(keep):
        return turns

    result = list(turns)  # shallow copy for mutation

    # 짧은 턴을 인접 턴에 흡수 (뒤에서부터 처리하여 인덱스 안정)
    for i in range(len(result) - 1, -1, -1):
        if keep[i]:
            continue
        short = result[i]
        text = short.text.strip()
        if not text:
            result.pop(i)
            keep.pop(i)
            continue

        # 직전 턴 우선, 없으면 직후 턴에 흡수
        if i > 0 and keep[i - 1]:
            result[i - 1].end = max(result[i - 1].end, short.end)
            result[i - 1].text = f"{result[i - 1].text} {text}".strip()
        elif i < len(result) - 1 and keep[i + 1]:
            result[i + 1].start = min(result[i + 1].start, short.start)
            result[i + 1].text = f"{text} {result[i + 1].text}".strip()
        else:
            # 양쪽 모두 짧은 턴 → 그냥 유지 (연속 짧은 턴은 드묾)
            continue

        result.pop(i)
        keep.pop(i)

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
    return _filter_short_turns(merged, s.min_turn_sec, s.min_words_per_turn)
