import requests

base = "http://localhost:8000"
s = requests.post(f"{base}/v1/sessions/", json={"sample_rate": 16000, "channels": 1}).json()
ssid = s["ssid"]
pcm = b"\x00\x00" * (16000 * 2)
files = {"file": ("chunk.pcm", pcm, "application/octet-stream")}
r = requests.post(f"{base}/v1/sessions/{ssid}/chunk", params={"seq": 0}, files=files)
print(r.json())
print(requests.post(f"{base}/v1/sessions/{ssid}/finalize").json())
