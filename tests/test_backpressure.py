from core.config import Settings


def test_backpressure_thresholds():
    s = Settings(global_queue_limit=100, backpressure_slow_ratio=0.6, backpressure_pause_ratio=0.9)
    assert int(s.global_queue_limit * s.backpressure_slow_ratio) == 60
    assert int(s.global_queue_limit * s.backpressure_pause_ratio) == 90
