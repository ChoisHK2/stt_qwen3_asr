# GPU STT 통합 서버 (Qwen3-ASR + pyannote diarization)

FastAPI 기반 실시간 STT 서버. vLLM으로 Qwen3-ASR 모델을 서빙하고, pyannote로 화자분리를 수행합니다.

## 아키텍처

올인원 Docker 이미지 (`qwenllm/qwen3-asr` 기반):
- **하나의 컨테이너**에 Redis + vLLM(qwen-asr-serve) + FastAPI 통합
- `docker build` → `docker run` 한 번으로 전체 서비스 기동

```
┌─ Docker Container ──────────────────────────┐
│  Redis (6379)  ←  FastAPI (8000)  →  vLLM (8001)  │
│                     ↕                              │
│              pyannote diarization                   │
└─────────────────────────────────────────────┘
```

### 주요 모듈
- `api/`: FastAPI WS(`/v1/ws`) + REST fallback
- `core/`: 설정, 세션 서비스, diarization-ASR 매칭, merge 정책
- `audio/`: PCM16 전처리(증폭/노이즈제거/품질지표)
- `storage/`: Redis 기반 TTL/복구/idempotency 저장소
- `clients/`: ASR(chat completions) / diarization 클라이언트

### vLLM 자체 큐잉 활용
vLLM은 continuous batching으로 동시 요청을 자체 관리합니다.
별도의 Redis 큐/워커 없이, API에서 직접 vLLM HTTP API를 호출합니다.
`MAX_CONCURRENT_ASR` (세마포어)와 vLLM의 `--max-num-seqs`로 동시 처리량을 제어합니다.

## 빠른 시작

### 사전 요구사항
- NVIDIA GPU 드라이버 + [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) 설치
- 중국 대륙에서 Docker Hub 접근이 느린 경우 registry mirror 설정 권장

### 1. 모델 다운로드
```bash
./scripts/download_models.sh ./models
```

### 2. 이미지 빌드 + 실행

**개발용 (0.6B)**
```bash
cp .env.dev .env
docker build -t qwen3-stt .
docker run --gpus all --env-file .env \
  -v ./models:/models \
  -p 8000:8000 \
  qwen3-stt
```

**프로덕션 (1.7B)**
```bash
cp .env.prod .env
docker build -t qwen3-stt .
docker run --gpus all --env-file .env \
  -v ./models:/models \
  -p 8000:8000 \
  qwen3-stt
```

컨테이너가 뜨면 Redis → vLLM → FastAPI 순서로 자동 기동됩니다.
vLLM이 준비될 때까지 대기 후 API가 시작됩니다.

> **참고**: `qwenllm/qwen3-asr:latest` 공식 이미지를 base로 사용합니다.
> 오디오 처리에 필요한 모든 의존성(soundfile, librosa, vLLM 등)이 포함되어 있습니다.

### (선택) docker-compose 멀티 서비스 모드
GPU를 분리하거나 스케일링이 필요한 경우:
```bash
docker compose --profile dev up -d --build   # 또는 --profile prod
```

## 모델 디렉토리 구조
```
models/
├── Qwen3-ASR-0.6B/          # 개발용 ASR 모델
├── Qwen3-ASR-1.7B/          # 프로덕션 ASR 모델
└── pyannote/
    └── speaker-diarization-community-1/
        ├── config.yaml       # 상대경로 참조 (Docker/로컬 모두 호환)
        ├── segmentation-3.0/
        └── wespeaker-voxceleb-resnet34-LM/
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
`.env` 파일만 바꾸면 됩니다:
| 환경 | .env 파일 | VLLM_MODEL | MAX_CONCURRENT_ASR |
|------|----------|-----------|-------------------|
| dev (0.6B) | `.env.dev` | `Qwen/Qwen3-ASR-0.6B` | 4 |
| prod (1.7B) | `.env.prod` | `Qwen/Qwen3-ASR-1.7B` | 32 |

## 100 커넥션 안정화 (프로덕션)
- vLLM `--max-num-seqs 32`: 동시 추론 요청 수 (B200 MIG 30GB 기준)
- `--gpu-memory-utilization 0.90`: GPU 메모리 활용률
- `MAX_CONCURRENT_ASR=32`: API→vLLM 동시 요청 세마포어
- vLLM이 자체 continuous batching으로 요청을 큐잉하므로, 32개 이상 요청이 와도 순서대로 처리됨
- 5초 청크 × 100 커넥션 = 초당 ~20 요청, vLLM이 배칭으로 처리

## HTTPS 지원

기본은 HTTP. SSL 인증서를 마운트하면 HTTPS가 활성화됩니다.

```bash
# 인증서 준비 (cert/ 디렉토리에 key.pem, cert.pem)
docker run --gpus all --env-file .env \
  -v ./models:/models \
  -v ./cert:/cert \
  -e SSL_KEYFILE=/cert/key.pem \
  -e SSL_CERTFILE=/cert/cert.pem \
  -p 8000:8000 \
  qwen3-stt
```

**HTTPS 영향 범위:**
- 외부 접속 (UI, REST, WebSocket): `https://ip:8000`, `wss://ip:8000/v1/ws`
- 컨테이너 내부 (FastAPI→vLLM, FastAPI→Redis): HTTP 유지, 영향 없음

> 자체서명 인증서 생성: `openssl req -x509 -nodes -days 365 -newkey rsa:2048 -keyout cert/key.pem -out cert/cert.pem`

## 포트 변경

기본 포트(8000)가 사용 중이면 `.env`에서 `APP_PORT`를 변경하고 `-p` 옵션도 맞춰줍니다:
```bash
# .env에서 APP_PORT=8010 설정 후:
docker run --gpus all --env-file .env -v ./models:/models -p 8010:8010 qwen3-stt
```

## 브라우저 UI 테스트
```
http://localhost:8000/ui    (HTTP)
https://localhost:8000/ui   (HTTPS 설정 시)
```
- 오프라인 회의: 마이크 녹음
- 온라인 회의: 마이크 + 시스템 오디오 (화면 공유)
- 파일 테스트: 오디오 파일 업로드

## diarization 토큰/오프라인 운영
- 로컬 모델이 있으면 토큰 없이 오프라인으로 동작
- 권장 절차:
  1. 온라인 PC에서 1회 라이선스 동의 + 토큰 발급
  2. `PYANNOTE_TOKEN=hf_xxx ./scripts/download_models.sh ./models` 실행
  3. 생성된 `./models/pyannote/speaker-diarization-community-1` 폴더를 오프라인 서버로 함께 이관
- 런타임에는 `PYANNOTE_LOCAL_PATH`가 존재하면 해당 로컬 경로를 우선 사용

## 오프라인 배포
```bash
# 온라인 PC에서
docker build -t qwen3-stt:latest .
docker save qwen3-stt:latest -o qwen3-stt.tar

# 오프라인 서버에서
docker load -i qwen3-stt.tar
cp .env.prod .env
docker run --gpus all --env-file .env -v ./models:/models -p 8000:8000 qwen3-stt
```

### 오프라인에서 코드만 수정 후 재빌드
이미 빌드된 `qwen3-stt:latest` 이미지 위에 패치 이미지를 만들 수 있습니다:
```dockerfile
# Dockerfile.fix
FROM qwen3-stt:latest
COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh
# 필요시 다른 파일도 COPY
```
```bash
docker build -f Dockerfile.fix -t qwen3-stt:fixed .
```
베이스 이미지를 다시 받을 필요 없이 수 초 만에 빌드됩니다.

## MIG(B200 30GB slice) 가이드
- `--max-num-seqs 32`로 시작, 메모리 부족 시 하향
- `chunk_sec=5`는 지연시간/품질 균형, `2`는 지연시간 유리
- `--gpu-memory-utilization 0.90` 보수적 설정 권장

## 테스트
```bash
pytest -q
python scripts/e2e_smoke.py
```
