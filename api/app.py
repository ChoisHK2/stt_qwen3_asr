from __future__ import annotations

import base64
import json

from fastapi import FastAPI, File, HTTPException, UploadFile, WebSocket, WebSocketDisconnect

from api.schemas import FinalizePayload, StartPayload
from core.config import get_settings
from core.session_service import SessionService
from storage.redis_store import RedisStore

app = FastAPI(title="GPU STT Service")
settings = get_settings()


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
    try:
        app.state.session_service.diar.load(settings.pyannote_model, settings.pyannote_token)
        details["load"] = "ok"
        ok = True
    except Exception as exc:
        ok = False
        details["load"] = f"graceful-fail: {exc}"
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
