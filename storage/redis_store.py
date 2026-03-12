from __future__ import annotations

import glob
import json
import logging
import os
import time
from typing import Any

import aiofiles
import aiofiles.os

from redis.asyncio import Redis

from core.config import get_settings

logger = logging.getLogger("qwen3-asr.store")


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

    def _partials_key(self, ssid: str) -> str:
        return f"session:{ssid}:partials"

    def _status_key(self, ssid: str) -> str:
        return f"session:{ssid}:status"

    async def create_or_touch_session(self, ssid: str, payload: dict[str, Any]) -> None:
        key = self._session_key(ssid)
        await self.redis.hset(key, mapping={k: json.dumps(v).encode() for k, v in payload.items()})
        await self.redis.expire(key, self.settings.session_ttl_sec)
        await self.redis.expire(self._seq_key(ssid), self.settings.session_ttl_sec)
        await self.redis.expire(self._partials_key(ssid), self.settings.session_ttl_sec)
        await self.redis.expire(self._status_key(ssid), self.settings.session_ttl_sec)

    async def get_session_meta(self, ssid: str) -> dict[str, Any]:
        data = await self.redis.hgetall(self._session_key(ssid))
        return {k.decode(): json.loads(v) for k, v in data.items()} if data else {}

    async def record_chunk(self, ssid: str, seq: int) -> bool:
        """Record chunk sequence number for dedup. Returns True if new."""
        added = await self.redis.sadd(self._seq_key(ssid), str(seq))
        return bool(added)

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

    # ── Disk-based PCM storage ──────────────────────────────────────

    def _pcm_len_key(self, ssid: str) -> str:
        return f"session:{ssid}:pcm_len"

    def _pcm_path(self, ssid: str) -> str:
        return os.path.join(self.settings.audio_data_dir, f"{ssid}.raw")

    async def append_pcm(self, ssid: str, pcm_bytes: bytes) -> int:
        """Append PCM data to disk file and return total length in bytes."""
        path = self._pcm_path(ssid)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        async with aiofiles.open(path, "ab") as f:
            await f.write(pcm_bytes)
        new_len = await self.redis.incrby(self._pcm_len_key(ssid), len(pcm_bytes))
        await self.redis.expire(self._pcm_len_key(ssid), self.settings.session_ttl_sec)
        return new_len

    async def get_pcm_length(self, ssid: str) -> int:
        """Return total PCM bytes stored (from Redis counter, no disk I/O)."""
        raw = await self.redis.get(self._pcm_len_key(ssid))
        return int(raw) if raw else 0

    async def get_pcm(self, ssid: str) -> bytes:
        """Read entire PCM from disk."""
        path = self._pcm_path(ssid)
        try:
            async with aiofiles.open(path, "rb") as f:
                return await f.read()
        except FileNotFoundError:
            return b""

    async def get_pcm_slice(self, ssid: str, start_byte: int, end_byte: int) -> bytes:
        """Read a byte range from the PCM file on disk."""
        path = self._pcm_path(ssid)
        try:
            async with aiofiles.open(path, "rb") as f:
                await f.seek(start_byte)
                return await f.read(end_byte - start_byte)
        except FileNotFoundError:
            return b""

    # ── Disk file cleanup ────────────────────────────────────────────

    async def delete_session_files(self, ssid: str) -> None:
        """세션의 .raw 및 .wav 파일을 삭제한다."""
        for ext in (".raw", ".wav"):
            path = os.path.join(self.settings.audio_data_dir, f"{ssid}{ext}")
            try:
                await aiofiles.os.remove(path)
                logger.info("Deleted %s", path)
            except FileNotFoundError:
                pass

    async def cleanup_orphan_files(self, max_age_sec: int | None = None) -> int:
        """Redis에 세션이 없는 orphan 오디오 파일을 삭제한다.

        max_age_sec가 지정되면, 그보다 오래된 파일만 삭제한다.
        기본값은 session_ttl_sec.
        """
        if max_age_sec is None:
            max_age_sec = self.settings.session_ttl_sec

        audio_dir = self.settings.audio_data_dir
        if not os.path.isdir(audio_dir):
            return 0

        now = time.time()
        deleted = 0

        for path in glob.glob(os.path.join(audio_dir, "*.raw")):
            ssid = os.path.basename(path).removesuffix(".raw")

            # 파일이 충분히 오래되었는지 확인
            try:
                mtime = os.path.getmtime(path)
            except OSError:
                continue
            if now - mtime < max_age_sec:
                continue

            # Redis에 세션이 남아 있는지 확인
            exists = await self.redis.exists(self._session_key(ssid))
            if exists:
                continue

            # orphan — 삭제
            for ext in (".raw", ".wav"):
                fpath = os.path.join(audio_dir, f"{ssid}{ext}")
                try:
                    await aiofiles.os.remove(fpath)
                    deleted += 1
                    logger.info("Orphan cleanup: deleted %s", fpath)
                except FileNotFoundError:
                    pass

        return deleted
