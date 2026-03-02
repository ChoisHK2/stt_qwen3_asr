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
        TimelineTurn("S2", 2.2, 2.8, "x"),
    ]
    out = merge_turns(turns)
    assert len(out) == 2
    assert out[0].text == "hello world"
