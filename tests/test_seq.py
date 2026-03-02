class FakeStore:
    def __init__(self):
        self.seen = set()

    def record_chunk(self, ssid, seq, payload):
        k = (ssid, seq)
        if k in self.seen:
            return False
        self.seen.add(k)
        return True


def test_seq_idempotent():
    fs = FakeStore()
    assert fs.record_chunk("a", 1, b"x") is True
    assert fs.record_chunk("a", 1, b"x") is False
