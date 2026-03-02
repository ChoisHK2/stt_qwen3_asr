# GPU STT 통합 서버 (Qwen3-ASR + pyannote diarization)

WAS는 chunk를 그대로 프록시하고, GPU 서버가 세션/큐/전처리/partial/finalize를 처리하는 레퍼런스 구현입니다.

## 아키텍처
- `api/`: FastAPI WS(`/v1/ws`) + REST fallback
- `core/`: 설정, 세션 서비스, diarization-ASR 매칭, merge 정책
- `audio/`: PCM16 전처리(증폭/노이즈제거/품질지표)
- `storage/`: Redis 기반 TTL/복구/idempotency 저장소
- `workers/`: 큐 소비 워커(추론 동시성 제한 지점)
- `clients/`: WS/REST 샘플 클라이언트

## 파일 트리
```text
.
├── Dockerfile
├── docker-compose.yml
├── entrypoint.sh
├── .env.example
├── pyproject.toml
├── README.md
├── docs/
│   └── FAQ.md
├── api/
│   ├── app.py
│   └── schemas.py
├── audio/
│   └── preprocess.py
├── clients/
│   ├── rest_client.py
│   └── ws_client.py
├── core/
│   ├── config.py
│   ├── matching.py
│   ├── models.py
│   └── session_service.py
├── scripts/
│   ├── download_models.sh
│   └── e2e_smoke.py
├── storage/
│   └── redis_store.py
├── workers/
│   └── chunk_worker.py
└── tests/
    ├── test_backpressure.py
    ├── test_merge.py
    └── test_seq.py
```

## API 요약
### WS `/v1/ws`
- start: `{type:"start", payload:{ssid?, sample_rate, channels, chunk_sec?}}`
- chunk: binary frame = `json-header + '\n' + base64(pcm16raw)`
- finalize: `{type:"finalize", payload:{ssid}}`
- status: `{type:"status", payload:{ssid}}`

응답 이벤트
- `ack`, `partial`, `final`, `error`

### REST fallback
- `POST /v1/sessions/`
- `POST /v1/sessions/{ssid}/chunk?seq=...&t0=...`
- `POST /v1/sessions/{ssid}/finalize`
- `GET /v1/sessions/{ssid}/status`
- `GET /__ping`

## 백프레셔 정책
- Redis global queue 길이 기반 `backlog_hint`: `ok | slow_down | paused`
- 큐 초과 시 REST는 `429`, WS는 `error`
- `SESSION_QUEUE_LIMIT`, `GLOBAL_QUEUE_LIMIT`, 비율(`BACKPRESSURE_*`)로 튜닝

## 모델 다운로드 (온라인 PC)
```bash
./scripts/download_models.sh ./models
```

## vLLM 실행 예시
```bash
docker run --rm --gpus all -p 8001:8001 -v $(pwd)/models:/models \
  vllm/vllm-openai:latest \
  --model /models/Qwen3-ASR-1.7B \
  --served-model-name Qwen/Qwen3-ASR-1.7B \
  --host 0.0.0.0 --port 8001 --max-num-seqs 8
```

## 로컬 실행
```bash
cp .env.example .env
docker compose up -d --build
curl localhost:8000/__ping
```

## 개인 PC Self-test
- `/v1/selftest/model`: import/모델 로드 가능성 확인, GPU 없으면 graceful degrade
- `/v1/selftest/diarization`: pyannote 파이프라인 로드 체크
- `/v1/selftest/pipeline`: 전처리~파이프라인 스모크 경로 확인

## 오프라인 배포
1. 온라인 PC에서 이미지 빌드 및 export
```bash
docker build -t gpu-stt:latest .
docker save gpu-stt:latest -o gpu-stt.tar
```
2. `gpu-stt.tar` + `models/`를 오프라인 GPU 서버로 이동
3. 서버에서 import
```bash
docker load -i gpu-stt.tar
docker run --rm --gpus all --network host -v /opt/models:/models gpu-stt:latest
```

## MIG(H200 ~20GB slice) 가이드
- `WORKER_CONCURRENCY=1~2`로 시작
- `chunk_sec=2`는 지연시간 유리, `4`는 처리량 유리
- `--max-num-seqs`, `gpu-memory-utilization` 보수적 설정 권장

## 테스트
```bash
pytest -q
python scripts/e2e_smoke.py
```
