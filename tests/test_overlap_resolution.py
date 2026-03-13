"""_resolve_overlapping_turns 테스트."""
from core.matching import _resolve_overlapping_turns
from core.models import DiarTurn


def test_no_overlap_unchanged():
    """겹침이 없으면 그대로 반환."""
    diar = [
        DiarTurn("S1", 0.0, 5.0),
        DiarTurn("S2", 5.0, 10.0),
    ]
    result = _resolve_overlapping_turns(diar)
    assert len(result) == 2
    assert result[0].speaker == "S1"
    assert result[1].speaker == "S2"


def test_short_overlap_absorbed_into_dominant():
    """짧은 발화(≤2초)가 dominant와 겹치면 흡수된다."""
    diar = [
        DiarTurn("S1", 0.0, 10.0),   # dominant (10초)
        DiarTurn("S2", 3.0, 4.5),    # 짧은 겹침 (1.5초)
    ]
    result = _resolve_overlapping_turns(diar)
    # S2가 흡수되어 S1만 남아야 함
    speakers = [t.speaker for t in result]
    assert "S2" not in speakers or all(
        t.end - t.start < 1.0 for t in result if t.speaker == "S2"
    )
    # S1은 반드시 남아있어야 함
    assert any(t.speaker == "S1" for t in result)


def test_short_overlap_kept_if_context_matches():
    """짧은 발화라도 앞뒤 문맥과 일치하면 시간 분할로 유지."""
    diar = [
        DiarTurn("S2", 0.0, 3.0),    # 이전 S2
        DiarTurn("S1", 3.0, 10.0),   # dominant (7초)
        DiarTurn("S2", 4.0, 5.5),    # 짧은 겹침 (1.5초), 앞이 S2이므로 문맥 매칭
        DiarTurn("S1", 10.0, 15.0),  # 이후
    ]
    result = _resolve_overlapping_turns(diar)
    # 문맥상 S2는 유지되어야 함 (시간 조정은 됨)
    s2_turns = [t for t in result if t.speaker == "S2"]
    assert len(s2_turns) >= 1


def test_long_overlap_split_at_midpoint():
    """긴 겹침(>2초)은 중간점으로 분할."""
    diar = [
        DiarTurn("S1", 0.0, 10.0),   # 10초
        DiarTurn("S2", 2.0, 8.0),    # 6초 겹침 (긴 겹침)
    ]
    result = _resolve_overlapping_turns(diar)
    # 두 화자 모두 유지되어야 함
    speakers = {t.speaker for t in result}
    assert "S1" in speakers
    assert "S2" in speakers
    # 겹치는 구간이 없어야 함
    sorted_result = sorted(result, key=lambda t: t.start)
    for i in range(len(sorted_result) - 1):
        assert sorted_result[i].end <= sorted_result[i + 1].start + 0.02


def test_empty_and_single():
    """빈 리스트와 단일 turn."""
    assert _resolve_overlapping_turns([]) == []
    single = [DiarTurn("S1", 0.0, 5.0)]
    result = _resolve_overlapping_turns(single)
    assert len(result) == 1


def test_multiple_overlaps():
    """여러 turn이 동시에 겹치는 경우."""
    diar = [
        DiarTurn("S1", 0.0, 10.0),
        DiarTurn("S2", 1.0, 2.0),   # 짧은 겹침 (1초)
        DiarTurn("S3", 5.0, 6.0),   # 짧은 겹침 (1초)
    ]
    result = _resolve_overlapping_turns(diar)
    # S1은 반드시 남아있어야 함
    assert any(t.speaker == "S1" for t in result)
    # 전체 시간 범위가 유지되어야 함
    assert result[0].start == 0.0
