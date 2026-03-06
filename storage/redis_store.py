from __future__ import annotations

import json
from typing import Any

from redis.asyncio import Redis

from core.config import get_settings


class RedisStore:
    def __init__(self, redis: Redis):
        self.redis = redis
        self.settings = get_settings()

    @classmethod
    async def from_url(cls, url: str) -> "RedisStore":
        return cls(Redis.from_url(url, decode_responses=False))

    def _session_key(self, ssid: str) -> str:
        return f"session:{ssid}:meta"

    def _seq_key(self, ssid: str) -> str:
        return f"session:{ssid}:seq"

    def _chunks_key(self, ssid: str) -> str:
        return f"session:{ssid}:chunks"

    def _partials_key(self, ssid: str) -> str:
        return f"session:{ssid}:partials"

    def _status_key(self, ssid: str) -> str:
        return f"session:{ssid}:status"

    async def create_or_touch_session(self, ssid: str, payload: dict[str, Any]) -> None:
        key = self._session_key(ssid)
        await self.redis.hset(key, mapping={k: json.dumps(v).encode() for k, v in payload.items()})
        await self.redis.expire(key, self.settings.session_ttl_sec)
        await self.redis.expire(self._seq_key(ssid), self.settings.session_ttl_sec)
        await self.redis.expire(self._chunks_key(ssid), self.settings.session_ttl_sec)
        await self.redis.expire(self._partials_key(ssid), self.settings.session_ttl_sec)
        await self.redis.expire(self._status_key(ssid), self.settings.session_ttl_sec)

    async def get_session_meta(self, ssid: str) -> dict[str, Any]:
        data = await self.redis.hgetall(self._session_key(ssid))
        return {k.decode(): json.loads(v) for k, v in data.items()} if data else {}

    async def update_session_field(self, ssid: str, field: str, value: Any) -> None:
        await self.redis.hset(self._session_key(ssid), field, json.dumps(value).encode())

    async def record_chunk(self, ssid: str, seq: int, payload: bytes) -> bool:
        seq_key = self._seq_key(ssid)
        added = await self.redis.sadd(seq_key, str(seq))
        if added:
            await self.redis.hset(self._chunks_key(ssid), str(seq), payload)
        return bool(added)

    async def get_chunk(self, ssid: str, seq: int) -> bytes | None:
        return await self.redis.hget(self._chunks_key(ssid), str(seq))

    async def set_status(self, ssid: str, mapping: dict[str, Any]) -> None:
        await self.redis.hset(
            self._status_key(ssid), mapping={k: json.dumps(v).encode() for k, v in mapping.items()}
        )

    async def get_status(self, ssid: str) -> dict[str, Any]:
        data = await self.redis.hgetall(self._status_key(ssid))
        return {k.decode(): json.loads(v) for k, v in data.items()} if data else {}

    async def append_partial(self, ssid: str, payload: dict[str, Any]) -> None:
        await self.redis.rpush(self._partials_key(ssid), json.dumps(payload).encode())

    async def get_partials(self, ssid: str) -> list[dict[str, Any]]:
        rows = await self.redis.lrange(self._partials_key(ssid), 0, -1)
        return [json.loads(r) for r in rows]

    async def queue_length(self) -> int:
        return await self.redis.llen("queue:chunks")

    async def enqueue_chunk(self, payload: dict[str, Any]) -> int:
        return await self.redis.rpush("queue:chunks", json.dumps(payload).encode())

    async def dequeue_chunk(self, timeout: int = 1):
        return await self.redis.blpop("queue:chunks", timeout=timeout)
