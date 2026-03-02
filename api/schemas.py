from pydantic import BaseModel


class StartPayload(BaseModel):
    ssid: str | None = None
    sample_rate: int
    channels: int = 1
    chunk_sec: int = 2


class ChunkMeta(BaseModel):
    ssid: str
    seq: int
    t0: float | None = None


class FinalizePayload(BaseModel):
    ssid: str


class StatusPayload(BaseModel):
    ssid: str
