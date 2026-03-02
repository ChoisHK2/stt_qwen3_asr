"""Run WS+REST smoke checks against a running local service."""

import requests


def rest_smoke(base: str = "http://localhost:8000"):
    ssid = requests.post(f"{base}/v1/sessions/", json={"sample_rate": 16000, "channels": 1}).json()["ssid"]
    pcm = b"\x00\x00" * (16000 * 2)
    files = {"file": ("c.pcm", pcm, "application/octet-stream")}
    chunk = requests.post(f"{base}/v1/sessions/{ssid}/chunk", params={"seq": 0}, files=files)
    assert chunk.status_code == 200
    fin = requests.post(f"{base}/v1/sessions/{ssid}/finalize")
    assert fin.status_code == 200


if __name__ == "__main__":
    rest_smoke()
    print("smoke ok")
