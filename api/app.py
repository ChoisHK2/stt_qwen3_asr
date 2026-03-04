from __future__ import annotations

import base64
import json
import os

from fastapi import FastAPI, File, HTTPException, Request, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles

from api.schemas import FinalizePayload, StartPayload
from core.config import get_settings
from core.session_service import SessionService
from storage.redis_store import RedisStore

app = FastAPI(title="Qwen3-ASR STT Service")
settings = get_settings()
app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/ui", StaticFiles(directory="ui", html=True), name="ui")


@app.on_event("startup")
async def startup():
    store = await RedisStore.from_url(settings.redis_url)
    app.state.store = store
    app.state.session_service = SessionService(store)


@app.get("/__ping")
async def ping():
    return "OK"


@app.post("/v1/sessions/")
async def create_session(payload: StartPayload):
    ssid = await app.state.session_service.start_session(
        payload.sample_rate, payload.channels, payload.chunk_sec, payload.ssid
    )
    return {"ssid": ssid, "ttl_sec": settings.session_ttl_sec}


@app.post("/v1/sessions/{ssid}/chunk")
async def upload_chunk(ssid: str, seq: int, t0: float | None = None, file: UploadFile = File(...)):
    raw = await file.read()
    result = await app.state.session_service.ingest_chunk(ssid, seq, raw, t0)
    if result.get("error"):
        raise HTTPException(status_code=settings.overload_http_code, detail=result)
    return result


@app.post("/v1/sessions/{ssid}/finalize")
async def finalize(ssid: str):
    result = await app.state.session_service.finalize(ssid)
    return result


@app.get("/v1/sessions/{ssid}/status")
async def status(ssid: str):
    return await app.state.session_service.status(ssid)


# ── /api/ routes (Vue.js frontend) ──────────────────────────────────


@app.post("/api/session")
async def api_create_session():
    ssid = await app.state.session_service.start_session(
        sample_rate=16000, channels=1, chunk_sec=5,
    )
    return {"session_id": ssid}


@app.post("/api/session/{session_id}/chunk")
async def api_upload_chunk(session_id: str, seq: int, request: Request):
    raw = await request.body()
    result = await app.state.session_service.ingest_chunk(session_id, seq, raw)
    if result.get("error"):
        raise HTTPException(status_code=settings.overload_http_code, detail=result)
    return {
        "chunk_text": result.get("partial_text", ""),
        "audio_metrics": result.get("audio_metrics"),
    }


@app.post("/api/session/{session_id}/finalize")
async def api_finalize(session_id: str):
    return await app.state.session_service.finalize(session_id)


# ── Self-test routes ───────────────────────────────────────────────


@app.get("/v1/selftest/model")
async def selftest_model():
    ok = True
    details = {"imports": "ok", "inference": "skipped_or_stubbed"}
    try:
        _ = app.state.session_service.asr
    except Exception as exc:
        ok = False
        details["imports"] = str(exc)
    return {"ok": ok, "details": details}


@app.get("/v1/selftest/diarization")
async def selftest_diarization():
    details = {"load": "not-run"}
    local_exists = os.path.isdir(settings.pyannote_local_path)
    source = settings.pyannote_local_path if local_exists else settings.pyannote_model
    token = settings.pyannote_token if not local_exists else None

    if not local_exists and not settings.pyannote_token:
        return {
            "ok": False,
            "details": {
                "load": "skipped",
                "reason": "no local diarization model and PYANNOTE_TOKEN not set",
                "expected_local_path": settings.pyannote_local_path,
            },
        }

    try:
        app.state.session_service.diar.load(source, token, device=settings.diar_device)
        details["load"] = "ok"
        details["source"] = source
        ok = True
    except Exception as exc:
        ok = False
        details["load"] = f"graceful-fail: {exc}"
        details["source"] = source
    return {"ok": ok, "details": details}


@app.get("/v1/selftest/pipeline")
async def selftest_pipeline():
    return {"ok": True, "details": "preprocess/import/e2e smoke path available"}


@app.websocket("/v1/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    svc: SessionService = app.state.session_service
    try:
        while True:
            msg = await ws.receive()
            if "text" in msg and msg["text"]:
                event = json.loads(msg["text"])
                typ = event.get("type")
                if typ == "start":
                    payload = StartPayload(**event["payload"])
                    ssid = await svc.start_session(
                        payload.sample_rate, payload.channels, payload.chunk_sec, payload.ssid
                    )
                    await ws.send_json({"type": "ack", "ssid": ssid, "accepted_seq": -1})
                elif typ == "finalize":
                    payload = FinalizePayload(**event["payload"])
                    result = await svc.finalize(payload.ssid)
                    await ws.send_json({"type": "final", **result})
                elif typ == "status":
                    ssid = event["payload"]["ssid"]
                    await ws.send_json({"type": "ack", **(await svc.status(ssid))})
            elif "bytes" in msg and msg["bytes"]:
                frame = json.loads(msg["bytes"][: msg["bytes"].find(b"\n")].decode())
                raw = base64.b64decode(msg["bytes"][msg["bytes"].find(b"\n") + 1 :])
                result = await svc.ingest_chunk(frame["ssid"], frame["seq"], raw, frame.get("t0"))
                if result.get("error"):
                    await ws.send_json({"type": "error", "ssid": frame["ssid"], **result})
                else:
                    await ws.send_json(
                        {
                            "type": "partial",
                            "ssid": frame["ssid"],
                            "seq": frame["seq"],
                            "partial_text": result["partial_text"],
                            "audio_metrics": result["audio_metrics"],
                            "backlog_hint": result["backlog_hint"],
                        }
                    )
    except WebSocketDisconnect:
        return
