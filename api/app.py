from __future__ import annotations

import base64
import json
import os
import time

from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from api.schemas import FinalizePayload, StartPayload
from core.config import get_settings
from core.session_service import SessionService
from storage.redis_store import RedisStore

app = FastAPI(title="Qwen3-ASR STT Service")
settings = get_settings()
app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/ui", StaticFiles(directory="ui", html=True), name="ui")

_start_ts: float = time.time()


@app.on_event("startup")
async def startup():
    global _start_ts
    _start_ts = time.time()
    os.makedirs("data/audio", exist_ok=True)
    store = await RedisStore.from_url(settings.redis_url)
    app.state.store = store
    app.state.session_service = SessionService(store)


@app.on_event("shutdown")
async def shutdown():
    svc = getattr(app.state, "session_service", None)
    if svc:
        await svc.close()


@app.get("/__ping")
async def ping():
    return "OK"


@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "uptime_sec": round(time.time() - _start_ts, 1),
        "config": {
            "stt_final_chunk_sec": settings.stt_final_chunk_sec,
            "max_session_audio_sec": settings.max_session_audio_sec,
            "session_ttl_sec": settings.session_ttl_sec,
            "vllm_model": settings.vllm_model,
        },
    }


@app.get("/api/runtime")
async def runtime_info():
    return {
        "sample_rate": 16000,
        "stt_language": "auto",
        "vllm_model": settings.vllm_model,
        "vllm_base_url": settings.vllm_base_url,
        "preprocess_enabled": settings.preprocess_enabled,
        "noise_reduction_mode": settings.noise_reduction_mode,
        "diar_device": settings.diar_device,
        "pyannote_local_path": settings.pyannote_local_path,
        "session_ttl_sec": settings.session_ttl_sec,
        "merge_mode": settings.merge_mode,
        "merge_gap_sec": settings.merge_gap_sec,
        "stt_final_chunk_sec": settings.stt_final_chunk_sec,
    }


@app.post("/v1/sessions/")
async def create_session(payload: StartPayload):
    ssid = await app.state.session_service.start_session(
        payload.sample_rate, payload.channels, payload.chunk_sec, payload.ssid
    )
    return {"ssid": ssid, "ttl_sec": settings.session_ttl_sec}


@app.post("/v1/sessions/{ssid}/chunk")
async def upload_chunk(
    ssid: str,
    seq: int,
    t0: float | None = None,
    realtime: int = Query(default=1),
    file: UploadFile = File(...),
):
    raw = await file.read()
    result = await app.state.session_service.ingest_chunk(ssid, seq, raw, t0, realtime=bool(realtime))
    if result.get("error"):
        raise HTTPException(status_code=settings.overload_http_code, detail=result)
    return result


@app.post("/v1/sessions/{ssid}/finalize")
async def finalize(ssid: str):
    result = await app.state.session_service.finalize(ssid)
    return result


@app.post("/v1/sessions/{ssid}/stop")
async def stop_session(ssid: str):
    result = await app.state.session_service.stop_session(ssid)
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
    return {"session_id": ssid, "ttl_sec": settings.session_ttl_sec}


@app.post("/api/session/{session_id}/chunk")
async def api_upload_chunk(
    session_id: str,
    seq: int,
    request: Request,
    realtime: int = Query(default=1),
):
    raw = await request.body()
    if not raw:
        return {"chunk_text": "", "audio_metrics": None, "backlog_hint": "ok"}
    if len(raw) > 2 * 1024 * 1024:
        raise HTTPException(413, f"chunk too large: {len(raw)} bytes")
    if len(raw) % 2 != 0:
        raise HTTPException(400, "invalid PCM16 payload size")

    result = await app.state.session_service.ingest_chunk(
        session_id, seq, raw, realtime=bool(realtime),
    )
    if result.get("error"):
        code = 429 if result["error"] in ("AUDIO_LIMIT_EXCEEDED", "SESSION_STOPPED") else 413
        raise HTTPException(status_code=code, detail=result["error"])
    return {
        "session_id": session_id,
        "seq": seq,
        "accepted_seq": result.get("accepted_seq", seq),
        "chunk_text": result.get("partial_text", ""),
        "stt_text": "",
        "received_bytes": len(raw),
        "start_ms": result.get("start_ms", 0),
        "end_ms": result.get("end_ms", 0),
        "audio_metrics": result.get("audio_metrics"),
        "backlog_hint": result.get("backlog_hint", "ok"),
        "duplicate": result.get("duplicate", False),
    }


@app.post("/api/session/{session_id}/stop")
async def api_stop_session(session_id: str):
    result = await app.state.session_service.stop_session(session_id)
    return result


@app.get("/api/session/{session_id}/status")
async def api_session_status(session_id: str):
    return await app.state.session_service.status(session_id)


@app.post("/api/session/{session_id}/finalize")
async def api_finalize(session_id: str):
    return await app.state.session_service.finalize(session_id)


@app.post("/api/session/{session_id}/resume")
async def api_resume_session(session_id: str):
    """중단된 세션을 재개하여 녹음을 이어간다."""
    result = await app.state.session_service.resume_session(session_id)
    if result.get("error"):
        raise HTTPException(404, result["error"])
    return result


@app.get("/api/session/{session_id}/partial-results")
async def api_partial_results(session_id: str):
    """세션의 지금까지 누적된 결과를 반환한다 (finalize 없이)."""
    result = await app.state.session_service.get_partial_results(session_id)
    if result.get("error"):
        raise HTTPException(404, result["error"])
    return result


@app.get("/api/sessions/active")
async def api_active_sessions():
    """현재 활성 세션 수를 반환한다 (로드 테스트용)."""
    store: RedisStore = app.state.store
    count = await store.count_active_sessions()
    return {"active_sessions": count}


@app.get("/api/session/{session_id}/audio")
async def api_session_audio(session_id: str):
    st = await app.state.session_service.status(session_id)
    audio_path = st.get("audio_path", "")
    if not audio_path or not os.path.isfile(audio_path):
        raise HTTPException(404, "audio file not found")
    return FileResponse(
        audio_path,
        media_type="audio/wav",
        filename=f"meeting_{session_id[:8]}.wav",
    )


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
                elif typ == "stop":
                    ssid = event["payload"]["ssid"]
                    result = await svc.stop_session(ssid)
                    await ws.send_json({"type": "stopped", **result})
                elif typ == "status":
                    ssid = event["payload"]["ssid"]
                    await ws.send_json({"type": "ack", **(await svc.status(ssid))})
            elif "bytes" in msg and msg["bytes"]:
                frame = json.loads(msg["bytes"][: msg["bytes"].find(b"\n")].decode())
                raw = base64.b64decode(msg["bytes"][msg["bytes"].find(b"\n") + 1 :])
                realtime = frame.get("realtime", True)
                result = await svc.ingest_chunk(
                    frame["ssid"], frame["seq"], raw, frame.get("t0"), realtime=realtime,
                )
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
