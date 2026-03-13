from core.config import get_settings
from core.matching import merge_turns
from core.models import TimelineTurn


def test_merge_gap():
    s = get_settings()
    s.merge_mode = "gap"
    s.merge_gap_sec = 0.5
    turns = [
        TimelineTurn("S1", 0.0, 1.0, "hello"),
        TimelineTurn("S1", 1.2, 2.0, "world"),
        TimelineTurn("S2", 2.2, 4.0, "hi there friend"),
    ]
    out = merge_turns(turns)
    assert len(out) == 2
    assert out[0].text == "hello world"
    assert out[1].speaker == "S2"


def test_merge_filters_short_turn():
    """min_turn_sec 미만 + min_words_per_turn 미만인 턴은 인접 턴에 흡수된다."""
    s = get_settings()
    s.merge_mode = "gap"
    s.merge_gap_sec = 0.5
    turns = [
        TimelineTurn("S1", 0.0, 2.0, "hello world there"),
        TimelineTurn("S2", 2.2, 2.8, "x"),  # 0.6초, 1단어 → 짧은 턴
        TimelineTurn("S1", 3.0, 5.0, "goodbye my friend"),
    ]
    out = merge_turns(turns)
    # S2 "x"는 짧아서 직전 S1에 흡수됨
    assert len(out) == 2
    assert "x" in out[0].text
    assert out[0].speaker == "S1"
