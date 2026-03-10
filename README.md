# GPU STT 서버 (Qwen3-ASR + pyannote diarization)

올인원 Docker 이미지. Redis + vLLM + FastAPI가 하나의 컨테이너에서 실행됩니다.

```
┌─ Container ─────────────────────────────────────┐
│  Redis (6379)  ←  FastAPI (8000)  →  vLLM (8001)│
│                      ↕                          │
│               pyannote diarization              │
└─────────────────────────────────────────────────┘
```

## 사전 요구사항

- NVIDIA GPU 드라이버 + [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)

## 빠른 시작

### 1. 모델 다운로드 (온라인 PC)
```bash
# pyannote 토큰이 필요하면: PYANNOTE_TOKEN=hf_xxx
./scripts/download_models.sh ./models
```

### 2. 빌드
```bash
docker build -t qwen3-stt .
```

### 3. 실행

**개발 (0.6B)**
```bash
cp .env.dev .env
docker run --gpus all --env-file .env -v ./models:/models -p 8000:8000 qwen3-stt
```

**프로덕션 (1.7B)**
```bash
cp .env.prod .env
docker run --gpus all --env-file .env -v ./models:/models -p 8000:8000 qwen3-stt
```

**특정 GPU 지정**
```bash
docker run --gpus "device=1" --env-file .env -v ./models:/models -p 8000:8000 qwen3-stt
```

**포트 변경** (8000이 사용 중일 때)
```bash
# .env에서 APP_PORT=8010으로 수정 후:
docker run --gpus all --env-file .env -v ./models:/models -p 8010:8010 qwen3-stt
```

**HTTPS**
```bash
# 인증서 생성 (자체서명):
# openssl req -x509 -nodes -days 365 -newkey rsa:2048 -keyout cert/key.pem -out cert/cert.pem

docker run --gpus all --env-file .env \
  -v ./models:/models -v ./cert:/cert \
  -e SSL_KEYFILE=/cert/key.pem -e SSL_CERTFILE=/cert/cert.pem \
  -p 8000:8000 qwen3-stt
```
HTTPS는 외부 접속(UI/REST/WebSocket)에만 적용됩니다. 컨테이너 내부 통신(FastAPI→vLLM)은 HTTP 유지.

## 오프라인 배포

```bash
# 온라인 PC
docker build -t qwen3-stt:latest .
docker save qwen3-stt:latest -o qwen3-stt.tar

# 오프라인 서버
docker load -i qwen3-stt.tar
cp .env.prod .env
docker run --gpus all --env-file .env -v ./models:/models -p 8000:8000 qwen3-stt
```

### 코드 패치 후 빠른 재빌드
```dockerfile
# Dockerfile.fix
FROM qwen3-stt:latest
COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh
```
```bash
docker build -f Dockerfile.fix -t qwen3-stt:fixed .   # 수 초 완료
```

## 모델 디렉토리 구조
```
models/
├── Qwen3-ASR-0.6B/
├── Qwen3-ASR-1.7B/
└── pyannote/
    └── speaker-diarization-community-1/
        ├── config.yaml
        ├── segmentation-3.0/
        └── wespeaker-voxceleb-resnet34-LM/
```

## API
- UI: `http://localhost:8000/ui`
- 헬스체크: `GET /__ping`
- 세션 생성: `POST /api/session`
- 청크 업로드: `POST /api/session/{id}/chunk?seq=0&realtime=1`
- 녹음 종료: `POST /api/session/{id}/stop`
- 최종 결과: `POST /api/session/{id}/finalize`
- 상태 확인: `GET /api/session/{id}/status`
- WebSocket: `WS /v1/ws`

## 테스트
```bash
pytest -q
python scripts/e2e_smoke.py
```
