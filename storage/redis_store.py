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

    # ── STT Final re-processing keys ───────────────────────────────

    def _stt_final_key(self, ssid: str) -> str:
        return f"session:{ssid}:stt_final"

    async def set_stt_final(self, ssid: str, items: list[dict[str, Any]]) -> None:
        key = self._stt_final_key(ssid)
        await self.redis.set(key, json.dumps(items).encode())
        await self.redis.expire(key, self.settings.session_ttl_sec)

    async def get_stt_final(self, ssid: str) -> list[dict[str, Any]]:
        raw = await self.redis.get(self._stt_final_key(ssid))
        if raw:
            return json.loads(raw)
        return []

    # ── Incremental diarization epochs ────────────────────────────

    def _diar_epochs_key(self, ssid: str) -> str:
        return f"session:{ssid}:diar_epochs"

    async def append_diar_epoch(self, ssid: str, epoch_data: dict[str, Any]) -> None:
        key = self._diar_epochs_key(ssid)
        await self.redis.rpush(key, json.dumps(epoch_data).encode())
        await self.redis.expire(key, self.settings.session_ttl_sec)

    async def get_diar_epochs(self, ssid: str) -> list[dict[str, Any]]:
        rows = await self.redis.lrange(self._diar_epochs_key(ssid), 0, -1)
        return [json.loads(r) for r in rows]

    # ── Active session counting ─────────────────────────────────────

    async def count_active_sessions(self) -> int:
        """현재 Redis에 남아 있는 활성 세션 수를 반환한다."""
        count = 0
        async for _ in self.redis.scan_iter(match="session:*:meta", count=100):
            count += 1
        return count

    # ── Full PCM storage (for final re-processing) ─────────────────

    def _pcm_key(self, ssid: str) -> str:
        return f"session:{ssid}:pcm"

    async def append_pcm(self, ssid: str, pcm_bytes: bytes) -> int:
        """Append PCM data and return total length in bytes."""
        key = self._pcm_key(ssid)
        await self.redis.append(key, pcm_bytes)
        length = await self.redis.strlen(key)
        await self.redis.expire(key, self.settings.session_ttl_sec)
        return length

    async def get_pcm(self, ssid: str) -> bytes:
        raw = await self.redis.get(self._pcm_key(ssid))
        return raw or b""
