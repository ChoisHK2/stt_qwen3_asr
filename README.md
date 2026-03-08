# GPU STT 통합 서버 (Qwen3-ASR + pyannote diarization)

FastAPI 기반 실시간 STT 서버. vLLM으로 Qwen3-ASR 모델을 서빙하고, pyannote로 화자분리를 수행합니다.

## 아키텍처
- `api/`: FastAPI WS(`/v1/ws`) + REST fallback
- `core/`: 설정, 세션 서비스, diarization-ASR 매칭, merge 정책
- `audio/`: PCM16 전처리(증폭/노이즈제거/품질지표)
- `storage/`: Redis 기반 TTL/복구/idempotency 저장소
- `clients/`: ASR/diarization 클라이언트 + WS/REST 샘플

### vLLM 자체 큐잉 활용
vLLM은 continuous batching으로 동시 요청을 자체 관리합니다.
별도의 Redis 큐/워커 없이, API에서 직접 vLLM HTTP API를 호출합니다.
`MAX_CONCURRENT_ASR` (세마포어)와 vLLM의 `--max-num-seqs`로 동시 처리량을 제어합니다.

## 파일 트리
```text
.
├── Dockerfile
├── docker-compose.yml          # --profile dev / prod
├── entrypoint.sh
├── .env.example
├── .env.dev                    # 개발용 (0.6B)
├── .env.prod                   # 프로덕션 (1.7B)
├── api/
│   ├── app.py
│   └── schemas.py
├── audio/
│   └── preprocess.py
├── clients/
│   ├── asr_client.py           # vLLM HTTP 클라이언트 (커넥션 풀 + 세마포어)
│   ├── diarization_client.py
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
├── ui/
│   └── index.html
└── tests/
    ├── test_backpressure.py
    ├── test_asr_client.py
    ├── test_merge.py
    └── test_seq.py
```

## 빠른 시작

### 개발용 (개인 PC, 0.6B 모델)
```bash
cp .env.dev .env
docker compose --profile dev up -d --build
```

### 프로덕션 (B200 MIG 30GB, 1.7B 모델)
```bash
cp .env.prod .env
docker compose --profile prod up -d --build
```

### 모델 다운로드 (온라인 PC)
```bash
./scripts/download_models.sh ./models
```

## API 요약
### WS `/v1/ws`
- start: `{type:"start", payload:{ssid?, sample_rate, channels, chunk_sec?}}`
- chunk: binary frame = `json-header + '\n' + base64(pcm16raw)`
- finalize: `{type:"finalize", payload:{ssid}}`
- status: `{type:"status", payload:{ssid}}`

응답 이벤트: `ack`, `partial`, `final`, `error`

### REST fallback
- `POST /v1/sessions/`
- `POST /v1/sessions/{ssid}/chunk?seq=...&t0=...`
- `POST /v1/sessions/{ssid}/finalize`
- `GET /v1/sessions/{ssid}/status`
- `GET /__ping`

### 프론트엔드 API (`/api/`)
- `POST /api/session` → 세션 생성
- `POST /api/session/{id}/chunk?seq=...&realtime=1` → 청크 업로드
- `POST /api/session/{id}/stop` → 녹음 종료 (백그라운드 STT 재처리 + 화자분리 시작)
- `POST /api/session/{id}/finalize` → 최종 결과 조회
- `GET /api/session/{id}/status` → 상태 확인

## 모델 전환 (0.6B ↔ 1.7B)
환경변수만 바꾸면 됩니다:
| 환경 | VLLM_MODEL | VLLM_BASE_URL | MAX_CONCURRENT_ASR |
|------|-----------|---------------|-------------------|
| dev (0.6B) | `Qwen/Qwen3-ASR-0.6B` | `http://vllm-dev:8001` | 4 |
| prod (1.7B) | `Qwen/Qwen3-ASR-1.7B` | `http://vllm-prod:8001` | 32 |

## 100 커넥션 안정화 (프로덕션)
- vLLM `--max-num-seqs 32`: 동시 추론 요청 수 (B200 MIG 30GB 기준)
- `--gpu-memory-utilization 0.90`: GPU 메모리 활용률
- `MAX_CONCURRENT_ASR=32`: API→vLLM 동시 요청 세마포어
- vLLM이 자체 continuous batching으로 요청을 큐잉하므로, 32개 이상 요청이 와도 순서대로 처리됨
- 5초 청크 × 100 커넥션 = 초당 ~20 요청, vLLM이 배칭으로 처리

## 브라우저 UI 테스트
```
http://localhost:8000/ui
```
- 오프라인 회의: 마이크 녹음
- 온라인 회의: 마이크 + 시스템 오디오 (화면 공유)
- 파일 테스트: 오디오 파일 업로드

## diarization 토큰/오프라인 운영
- 로컬 모델이 있으면 토큰 없이 오프라인으로 동작
- 권장 절차:
  1. 온라인 PC에서 1회 라이선스 동의 + 토큰 발급
  2. `PYANNOTE_TOKEN=hf_xxx ./scripts/download_models.sh ./models` 실행
  3. 생성된 `./models/pyannote-speaker-diarization-community-1` 폴더를 오프라인 서버로 함께 이관
- 런타임에는 `PYANNOTE_LOCAL_PATH`가 존재하면 해당 로컬 경로를 우선 사용

## 오프라인 배포
```bash
# 온라인 PC에서
docker build -t gpu-stt:latest .
docker save gpu-stt:latest -o gpu-stt.tar

# 오프라인 서버에서
docker load -i gpu-stt.tar
docker run --rm --gpus all --network host -v /opt/models:/models gpu-stt:latest
```

## MIG(B200 30GB slice) 가이드
- `--max-num-seqs 32`로 시작, 메모리 부족 시 하향
- `chunk_sec=5`는 지연시간/품질 균형, `2`는 지연시간 유리
- `--gpu-memory-utilization 0.90` 보수적 설정 권장

## 테스트
```bash
pytest -q
python scripts/e2e_smoke.py
```
