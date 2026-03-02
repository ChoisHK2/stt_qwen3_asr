from __future__ import annotations

import asyncio
import json

from core.config import get_settings
from storage.redis_store import RedisStore


async def worker_loop() -> None:
    settings = get_settings()
    store = await RedisStore.from_url(settings.redis_url)
    while True:
        item = await store.dequeue_chunk(timeout=1)
        if not item:
            await asyncio.sleep(0.05)
            continue
        _, payload = item
        task = json.loads(payload)
        # Placeholder for GPU-limited worker pool orchestration.
        # In production: move partial inference from API thread to workers.
        await store.set_status(task["ssid"], {"worker_seen_seq": task["seq"]})


if __name__ == "__main__":
    asyncio.run(worker_loop())
