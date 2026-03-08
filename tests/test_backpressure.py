from core.config import Settings


def test_max_concurrent_asr_default():
    s = Settings()
    assert s.max_concurrent_asr == 32


def test_max_concurrent_asr_custom():
    s = Settings(max_concurrent_asr=8)
    assert s.max_concurrent_asr == 8
