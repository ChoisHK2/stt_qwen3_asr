import asyncio
import base64
import json

import websockets


async def main():
    uri = "ws://localhost:8000/v1/ws"
    async with websockets.connect(uri, max_size=10_000_000) as ws:
        await ws.send(json.dumps({"type": "start", "payload": {"sample_rate": 16000, "channels": 1}}))
        ack = json.loads(await ws.recv())
        ssid = ack["ssid"]
        pcm = b"\x00\x00" * (16000 * 2)
        header = json.dumps({"ssid": ssid, "seq": 0}).encode() + b"\n"
        frame = header + base64.b64encode(pcm)
        await ws.send(frame)
        print(await ws.recv())
        await ws.send(json.dumps({"type": "finalize", "payload": {"ssid": ssid}}))
        print(await ws.recv())


if __name__ == "__main__":
    asyncio.run(main())
