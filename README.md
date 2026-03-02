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
docker run --rm --gpus all -p 8001:8001 \
  -e HF_HUB_OFFLINE=1 \
  -e TRANSFORMERS_OFFLINE=1 \
  -e HF_DATASETS_OFFLINE=1 \
  -v $(pwd)/models:/models \
  vllm/vllm-openai:latest \
  /models/Qwen3-ASR-1.7B \
  --served-model-name Qwen/Qwen3-ASR-1.7B \
  --host 0.0.0.0 --port 8001 \
  --gpu-memory-utilization 0.80 \
  --max-model-len 4096 \
  --max-num-seqs 1 \
  --max-num-batched-tokens 512 \
  --enforce-eager
```

### 운영 권장안
1. 온라인 PC에서 공통 산출물 준비
   - `./scripts/download_models.sh ./models`
   - `gpu-stt`(api/worker) 이미지 빌드+tar export
2. 개인 PC(ROCm)에서 사전검증
   - 가능하면 ROCm용 vLLM으로 `/v1/audio/transcriptions` 응답 확인
   - 어려우면 `/v1/selftest/*` + WS/REST ingest/finalize smoke로 기능 검증
3. 오프라인 서버(CUDA) 배포
   - CUDA vLLM 이미지 + 동일한 `models/` + 동일한 api/worker 이미지 사용

### 주의사항
- ROCm에서 통과한 성능 수치(지연/처리량)를 CUDA에 그대로 대입하면 오차가 큽니다.
- 따라서 성능튜닝은 반드시 최종 CUDA(MIG) 환경에서 재측정하세요.
- 기능 검증(프로토콜/복구/큐/품질지표)은 ROCm에서도 충분히 선행 가능합니다.

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
> 참고: 위 `gpu-stt:latest`는 API/worker 서비스 이미지입니다. vLLM 이미지는 대상 GPU 스택(CUDA/ROCm)에 맞는 태그를 별도로 관리하세요.

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
