from __future__ import annotations

from collections import defaultdict

from core.config import get_settings
from core.models import ASRSegment, DiarTurn, TimelineTurn


def _overlap(a0: float, a1: float, b0: float, b1: float) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


def map_speakers(segments: list[ASRSegment], diar: list[DiarTurn]) -> list[TimelineTurn]:
    settings = get_settings()
    turns: list[TimelineTurn] = []
    for seg in segments:
        votes = defaultdict(float)
        if seg.words:
            for w in seg.words:
                w0 = float(w.get("start", seg.start))
                w1 = float(w.get("end", seg.end))
                for d in diar:
                    ov = _overlap(w0, w1, d.start, d.end)
                    if ov > 0:
                        votes[d.speaker] += ov
        if not votes and settings.matching_fallback == "segment_majority":
            for d in diar:
                ov = _overlap(seg.start, seg.end, d.start, d.end)
                if ov > 0:
                    votes[d.speaker] += ov
        speaker = max(votes, key=votes.get) if votes else "UNKNOWN"
        turns.append(TimelineTurn(speaker=speaker, start=seg.start, end=seg.end, text=seg.text))
    return merge_turns(turns)


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
