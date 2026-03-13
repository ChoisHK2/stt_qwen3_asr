# STT Service 시스템 아키텍처

> **Qwen3-ASR + pyannote 기반 실시간 음성인식 & 화자분리 서비스**
>
> Confluence 게시용 문서

---

## 1. 시스템 개요

### 1.1 서비스 목적

온라인/오프라인 회의 녹음을 실시간으로 텍스트 변환(STT)하고, 화자를 자동으로 구분(Diarization)하여 화자별 타임라인을 생성하는 GPU 기반 AI 서비스입니다.

### 1.2 핵심 기능

| 기능 | 설명 |
|------|------|
| **실시간 STT** | 오디오 청크 수신 즉시 텍스트 변환 (부분 결과) |
| **고품질 STT 재처리** | 녹음 종료 후 60초 단위로 재처리하여 정확도 향상 |
| **화자 분리** | pyannote 모델로 화자별 발화 구간 자동 인식 |
| **인크리멘탈 화자 분리** | 10분 단위로 점진적 화자 분리 + 임베딩 기반 화자 매칭 |
| **겹침 발화 해소** | 동시 발화(overlapping speech) 감지 시 문맥 기반 화자 배정 |
| **화자-텍스트 매핑** | STT 결과와 화자 분리 결과를 시간축 기반으로 정합 |
| **오디오 전처리** | DC 제거, 게인 조정, 노이즈 감소, 품질 지표 산출 |
| **세션 관리** | 세션 생성/중단/재개/복구, 최대 2시간 녹음, 4시간 TTL |

### 1.3 기술 스택

| 계층 | 기술 |
|------|------|
| **AI 모델 (STT)** | Qwen3-ASR-0.6B / 1.7B (vLLM 서빙) |
| **AI 모델 (화자)** | pyannote/speaker-diarization-community-1 |
| **추론 엔진** | vLLM (continuous batching, GPU) |
| **웹 프레임워크** | FastAPI (async) + Uvicorn |
| **세션 저장소** | Redis (인메모리) |
| **오디오 저장** | 로컬 디스크 (PCM → WAV) |
| **컨테이너** | Docker (All-in-One) |
| **프론트엔드** | Vue.js 3 SPA (데모 UI) |

---

## 2. 전체 아키텍처

### 2.1 컨테이너 내부 구조

```
┌─────────────────────────────────────────────────────────┐
│                    Docker Container                      │
│                                                         │
│  ┌─────────────┐    ┌──────────────┐    ┌────────────┐ │
│  │   Redis      │    │   vLLM       │    │  pyannote   │ │
│  │   :6379      │    │   :8001      │    │  (in-proc)  │ │
│  │             │    │  Qwen3-ASR   │    │  Diarize    │ │
│  │  세션 메타    │    │  GPU 추론     │    │  CPU/GPU    │ │
│  └──────┬──────┘    └──────┬───────┘    └──────┬─────┘ │
│         │                  │                    │       │
│         └──────────┬───────┴────────────┬──────┘       │
│                    │                    │               │
│              ┌─────┴────────────────────┴─────┐        │
│              │         FastAPI :8000            │        │
│              │                                 │        │
│              │  ┌───────────┐ ┌─────────────┐ │        │
│              │  │ REST API   │ │ WebSocket    │ │        │
│              │  │ /api/*     │ │ /v1/ws       │ │        │
│              │  └─────┬─────┘ └──────┬──────┘ │        │
│              │        └───────┬──────┘        │        │
│              │                │               │        │
│              │  ┌─────────────┴────────────┐  │        │
│              │  │    SessionService         │  │        │
│              │  │                           │  │        │
│              │  │  ┌────────┐ ┌──────────┐ │  │        │
│              │  │  │ASR     │ │Diarize   │ │  │        │
│              │  │  │Client  │ │Client    │ │  │        │
│              │  │  └────────┘ └──────────┘ │  │        │
│              │  │  ┌────────┐ ┌──────────┐ │  │        │
│              │  │  │Audio   │ │Matching  │ │  │        │
│              │  │  │Preproc │ │Engine    │ │  │        │
│              │  │  └────────┘ └──────────┘ │  │        │
│              │  └──────────────────────────┘  │        │
│              └────────────────────────────────┘        │
│                                                         │
│  ┌─────────────────────┐                               │
│  │  data/audio/         │  ← PCM/WAV 파일 저장          │
│  │  {ssid}.raw          │                               │
│  │  {ssid}.wav          │                               │
│  └─────────────────────┘                               │
└─────────────────────────────────────────────────────────┘
         ▲
         │ :8000
         │
    ─────┴──────
    외부 네트워크
```

### 2.2 외부 연동 구조 (프론트엔드 ↔ 백엔드 ↔ STT 컨테이너)

```
┌────────────┐      ┌──────────────┐      ┌─────────────────┐
│            │      │              │      │                 │
│  Frontend  │ ───▶ │   Backend    │ ───▶ │ STT Container   │
│  (Browser) │      │  (Spring 등) │      │  (GPU Server)   │
│            │      │              │      │                 │
│  마이크 캡처 │      │  프록시/인증   │      │  Qwen3-ASR      │
│  PCM16 변환 │      │  세션 관리    │      │  pyannote       │
│  UI 렌더링  │      │  비즈니스 로직 │      │  Redis          │
│            │      │              │      │                 │
└────────────┘      └──────────────┘      └─────────────────┘

  [사용자 브라우저]     [사내 서버]          [GPU 서버/클라우드]
```

---

## 3. 데이터 흐름

### 3.1 실시간 녹음 → STT 흐름

```
① 브라우저 마이크       ② 프론트엔드          ③ 백엔드 서버         ④ STT 컨테이너
   캡처                  PCM 변환               프록시               처리

  getUserMedia()    →  Float32 → Int16    →  POST /chunk       →  PCM 수신
  AudioContext         다운샘플 16kHz          (raw bytes)          │
  ScriptProcessor      5초 버퍼링                                   ├─ 디스크 저장
                       ArrayBuffer                                 ├─ 전처리
                                                                   ├─ vLLM 추론
                                                                   │   (Qwen3-ASR)
                                                                   ▼
                                                              partial_text
                                                              + audio_metrics
                       chunk_text 표시    ◀  JSON 응답      ◀  응답 반환
```

### 3.2 녹음 종료 → 최종 결과 흐름

```
① Stop 요청                    ② 백그라운드 처리              ③ Finalize
   POST /stop                     (자동 시작)                   POST /finalize

   녹음 중단 ──────────▶ ┌── STT 재처리 (60초 단위) ──▶ ┌── 겹침 발화 해소
                         │   전체 오디오를 60초씩         │   ASR+Diar 정합
                         │   분할하여 고품질 추론         │   시간축 매핑
                         │                              │   화자별 텍스트 배정
                         ├── 화자 분리 ──────────────▶   │   갭 채우기
                         │   pyannote 파이프라인          │   연속 구간 병합
                         │   화자 임베딩 + 클러스터링     ▼
                         └──────────────────────────  timeline[]
                                                       full_text
          status 폴링                                   diarization[]
          (2~3초 간격)
          stt_done? diar_done?

   * finalize 호출 시 미완료 백그라운드 태스크가 있으면
     자동 대기 (timeout: asr_timeout_sec × 2)
```

### 3.3 인크리멘탈 화자 분리

장시간 녹음(10분+)에서 녹음 중에도 점진적으로 화자 분리를 수행합니다.

```
시간 ──────────────────────────────────────────────▶

  0min          10min          20min          30min
  │──── epoch 0 ──│──── epoch 1 ──│──── epoch 2 ──│
                  ▲               ▲               ▲
                  │               │               │
              diar 실행        diar 실행        diar 실행
              + 임베딩 추출    + 임베딩 추출    + 임베딩 추출
              (백그라운드)      (백그라운드)      (백그라운드)

  * 에폭 간 오버랩 없음 (임베딩 기반 매칭이므로 불필요)
  * 각 epoch의 화자 임베딩을 비교하여 동일 화자 매칭
  * cosine similarity 기반 (threshold: 0.55)
  * 동시 실행 제한: MAX_CONCURRENT_DIAR (기본 2)
```

---

## 4. 모듈 구조

### 4.1 디렉터리 구조

```
stt_qwen3_asr/
├── api/                    # API 계층
│   ├── app.py             # FastAPI 라우트 정의 (REST + WebSocket)
│   └── schemas.py         # Pydantic 요청/응답 모델
│
├── core/                   # 비즈니스 로직 계층
│   ├── config.py          # 환경변수 기반 설정 (Settings)
│   ├── models.py          # 데이터 클래스 (ChunkTask, AudioMetrics 등)
│   ├── session_service.py # 세션 관리 & 오케스트레이션
│   └── matching.py        # ASR-Diarization 시간축 정합 알고리즘
│
├── audio/                  # 오디오 전처리 계층
│   └── preprocess.py      # PCM→Float32, DC제거, 게인, 노이즈, 메트릭
│
├── clients/                # 외부 서비스 클라이언트
│   ├── asr_client.py      # vLLM ASR HTTP 클라이언트
│   ├── diarization_client.py  # pyannote 화자 분리
│   ├── rest_client.py     # REST API 예제 클라이언트
│   └── ws_client.py       # WebSocket 예제 클라이언트
│
├── storage/                # 저장소 계층
│   └── redis_store.py     # Redis 세션/메타데이터 관리
│
├── tests/                  # 테스트
├── static/                 # 프론트엔드 정적 파일
├── ui/                     # Vue.js SPA
├── docs/                   # 문서
└── scripts/                # 유틸리티 스크립트
```

### 4.2 모듈 의존성 다이어그램

```
          ┌───────────────┐
          │   api/app.py  │  ← HTTP/WS 엔드포인트
          └───────┬───────┘
                  │
                  ▼
        ┌─────────────────┐
        │ SessionService   │  ← 세션 생명주기 관리
        │                 │     청크 처리 오케스트레이션
        └──┬────┬────┬──┬┘
           │    │    │  │
     ┌─────┘    │    │  └──────┐
     ▼          ▼    ▼         ▼
┌─────────┐ ┌─────┐ ┌──────┐ ┌──────────┐
│ASR      │ │Audio│ │Diar  │ │Redis     │
│Client   │ │Prep │ │Client│ │Store     │
│         │ │     │ │      │ │          │
│vLLM HTTP│ │PCM  │ │pyan- │ │세션 메타  │
│→ 텍스트  │ │전처리│ │note  │ │PCM 디스크 │
└─────────┘ └─────┘ └──────┘ └──────────┘
     │                 │
     ▼                 ▼
  ┌──────┐       ┌──────────┐
  │vLLM  │       │pyannote  │
  │:8001 │       │(in-proc) │
  │GPU   │       │CPU/GPU   │
  └──────┘       └──────────┘
```

---

## 5. 핵심 컴포넌트 상세

### 5.1 SessionService (세션 서비스)

| 기능 | 메서드 | 설명 |
|------|--------|------|
| 세션 생성 | `start_session()` | UUID 발급, Redis에 메타데이터 저장 |
| 청크 수신 | `ingest_chunk()` | seq 중복 체크 → 디스크 저장 → 전처리 → 실시간 STT |
| 녹음 중단 | `stop_session()` | WAV 생성, 백그라운드 STT+Diar 시작 |
| 결과 생성 | `finalize()` | 백그라운드 태스크 대기 → STT + Diar 정합 → timeline 생성 |
| 상태 조회 | `status()` | 처리 진행 상태 반환 |
| 세션 재개 | `resume_session()` | 중단된 세션 녹음 재개 |
| 부분 결과 | `get_partial_results()` | 중간 결과 반환 |

**동시성 제어**:
- ASR 요청: `asyncio.Semaphore(MAX_CONCURRENT_ASR)` — 기본 32
- Diarization: `asyncio.Semaphore(MAX_CONCURRENT_DIAR)` — 기본 2

**백그라운드 태스크 관리**:
- `_track_task()`: 세션별 비동기 태스크 등록 (STT 재처리, 인크리멘탈 Diar)
- `cancel_session_tasks()`: 세션 취소 시 태스크 정리
- `finalize()` 호출 시 미완료 태스크 자동 대기 (timeout: `asr_timeout_sec × 2`)

### 5.2 ASR Client (음성 인식)

vLLM의 OpenAI-compatible API를 호출합니다.

```
PCM Float32 → WAV 변환 → Base64 인코딩 → vLLM /v1/chat/completions
                                              │
                                              ▼
                                    Qwen3-ASR 모델 추론
                                              │
                                              ▼
                                    응답 파싱 (<|transcription|> 태그)
                                              │
                                              ▼
                                    ASRSegment(start, end, text)
```

### 5.3 Diarization Client (화자 분리)

pyannote 파이프라인을 in-process로 실행합니다.

```
WAV 오디오 → pyannote Pipeline → DiarTurn(speaker, start, end)[]
                                      │
                                      ▼
                               화자 임베딩 추출
                               (centroid embedding 또는 별도 추출)
                                      │
                                      ▼
                               epoch 간 화자 매칭
                               (cosine similarity ≥ 0.55)
```

**인크리멘탈 모드 (에폭 단위)**:
- `diarize_epoch()`: 단일 에폭 diarize + 임베딩 추출
- `match_speakers_by_embedding()`: 이전/현재 에폭 화자 임베딩 비교
- `extract_speaker_embeddings()`: 각 화자의 발화 구간에서 임베딩 벡터 추출 후 평균

### 5.4 Matching Engine (정합 엔진)

STT 텍스트와 화자 분리 구간을 시간축 기반으로 결합합니다.

```
STT 결과:    "안녕하세요 오늘 회의를..."   [0.0s ~ 5.0s]
Diar 결과:   SPEAKER_00                   [0.0s ~ 3.5s]
             SPEAKER_01                   [3.8s ~ 7.2s]

          ↓ 시간 오버랩 기반 매핑 ↓

Timeline:  SPEAKER_00: "안녕하세요 오늘"      [0.0s ~ 3.5s]
           SPEAKER_01: "회의를..."            [3.8s ~ 7.2s]
```

**정합 알고리즘 (6단계)**:

1. **겹침 발화 해소** (`_resolve_overlapping_turns`)
   - pyannote가 동시 발화를 감지하면 같은 시간대에 여러 화자 turn 반환
   - ≤ 2초 짧은 겹침: 앞뒤 문맥(이전/다음 화자)을 확인하여 dominant 화자에 흡수
   - 문맥상 유효한 짧은 발화는 시간 분할로 유지
   - \> 2초 긴 겹침: 겹침 구간 중간점으로 분할
2. **연속 동일 화자 Diar 세그먼트 병합** (`_merge_same_speaker_diar`)
3. **갭 구간 인접 화자에 할당** (`_fill_diar_gaps`)
4. **ASR 텍스트를 시간 비율로 화자에 분배** (`_distribute_words_by_time`)
5. **15초 초과 턴 분할** (`_split_long_turns`)
6. **최종 연속 동일 화자 병합** (`_final_merge_same_speaker`)

**STT final ↔ chunk 정렬** (`align_final_with_chunks`):
- STT 재처리 텍스트(고품질)를 실시간 chunk 시간 경계(고정밀)에 맞춰 재분할
- SequenceMatcher로 final 단어 ↔ chunk 단어를 정렬하여 시간 상속

### 5.5 Audio Preprocessor (오디오 전처리)

```
PCM16 bytes
    │
    ▼
PCM16 → Float32 변환 (÷32768)
    │
    ▼ (PREPROCESS_ENABLED=true 일 때)
DC Offset 제거 (평균값 차감)
    │
    ▼
게인 조정 + 리미터 (target: -20 dBFS, limit: -1 dBFS)
    │
    ▼
노이즈 감소 (FAST: 3-tap, QUALITY: 9-tap)
    │
    ▼
품질 메트릭 산출 (RMS, Peak, SNR, 클리핑, 제안사항)
    │
    ▼
전처리된 Float32 + AudioMetrics
```

---

## 6. 데이터 저장 구조

### 6.1 Redis 키 구조

| 키 패턴 | 타입 | TTL | 설명 |
|---------|------|-----|------|
| `sess:{ssid}:meta` | Hash | 4h | 세션 메타 (sample_rate, channels, status 등) |
| `sess:{ssid}:seq` | Set | 4h | 수신된 seq 번호 (중복 체크) |
| `sess:{ssid}:partial:{seq}` | String | 4h | 청크별 부분 STT 결과 |
| `sess:{ssid}:stt_final` | String | 4h | 재처리 STT 결과 (JSON) |
| `sess:{ssid}:diar_epochs` | String | 4h | 인크리멘탈 Diar 결과 (에폭별 turn + 임베딩) |
| `sess:{ssid}:status` | String | 4h | 처리 상태 (recording/stopped/done) |

### 6.2 디스크 파일

| 경로 | 설명 |
|------|------|
| `data/audio/{ssid}.raw` | 원본 PCM16 바이트 (청크 append) |
| `data/audio/{ssid}.wav` | WAV 파일 (stop 시 생성) |

**스토리지 계산**:
- 1시간 오디오 ≈ 115 MB (PCM16, 16kHz, mono)
- 2시간 (최대) ≈ 230 MB

---

## 7. 환경 설정

### 7.1 주요 환경변수

| 변수 | 기본값 | 설명 |
|------|--------|------|
| `APP_HOST` | `0.0.0.0` | 바인드 주소 |
| `APP_PORT` | `8000` | FastAPI 포트 |
| `REDIS_URL` | `redis://localhost:6379/0` | Redis 연결 URL |
| `VLLM_MODEL` | `Qwen/Qwen3-ASR-0.6B` | ASR 모델명 |
| `VLLM_BASE_URL` | `http://localhost:8001` | vLLM API 주소 |
| `VLLM_GPU_UTIL` | `0.85` | GPU 메모리 사용률 |
| `MAX_CONCURRENT_ASR` | `32` | ASR 동시 요청 수 |
| `MAX_CONCURRENT_DIAR` | `2` | 화자분리 동시 실행 수 |
| `PREPROCESS_ENABLED` | `true` | 오디오 전처리 활성화 |
| `NOISE_REDUCTION_MODE` | `FAST` | 노이즈 감소 모드 (`FAST` / `QUALITY`) |
| `ENABLE_VAD` | `false` | VAD(Voice Activity Detection) 활성화 |
| `TARGET_RMS_DBFS` | `-20` | 게인 조정 목표 (dBFS) |
| `LIMITER_PEAK_DBFS` | `-1` | 리미터 피크 임계값 (dBFS) |
| `ASR_TIMEOUT_SEC` | `30` | ASR 요청 타임아웃 (초) |
| `MAX_SESSION_AUDIO_SEC` | `7200` | 세션당 최대 오디오 (초, 2시간) |
| `SESSION_TTL_SEC` | `14400` | 세션 만료 시간 (초, 4시간) |
| `STT_FINAL_CHUNK_SEC` | `60` | 재처리 세그먼트 크기 (초) |
| `DIAR_CHUNK_INTERVAL_SEC` | `600` | 인크리멘탈 Diar 주기 (초, 10분) |
| `DIAR_EMBEDDING_THRESHOLD` | `0.55` | 화자 임베딩 cosine similarity 임계값 |
| `DIAR_DEVICE` | `cpu` | 화자분리 디바이스 (`cpu` / `cuda` / `auto`) |
| `OVERLAP_POLICY` | `dominant` | 겹침 발화 처리 정책 |
| `MERGE_MODE` | `gap` | Diar 병합 모드 |
| `MERGE_GAP_SEC` | `0.35` | 병합 갭 임계값 (초) |
| `MIN_TURN_SEC` | `0.4` | 최소 턴 길이 (초) |
| `MAX_TURN_SEC` | `15` | 최대 턴 길이 (초, 초과 시 분할) |
| `MIN_WORDS_PER_TURN` | `1` | 턴 최소 단어 수 |
| `FINALIZE_ASYNC_THRESHOLD_SEC` | `480` | 비동기 finalize 임계값 (초) |
| `PARTIAL_MODE` | `on` | 실시간 부분 결과 모드 (`on` / `off`) |
| `OVERLOAD_HTTP_CODE` | `429` | 과부하 시 HTTP 상태 코드 |

### 7.2 배포 프로파일

| 프로파일 | 모델 | GPU 메모리 | 동시 세션 | 용도 |
|----------|------|------------|-----------|------|
| **dev** | Qwen3-ASR-0.6B | ~4 GB | ~4 | 개발/테스트 |
| **prod** | Qwen3-ASR-1.7B | ~8 GB | ~32 | 운영 (B200 MIG 30GB 권장) |

---

## 8. 배포 구조

### 8.1 All-in-One Docker

```bash
# 빌드
docker build -t qwen3-stt .

# 실행
docker run --gpus all \
  --env-file .env.prod \
  -v ./models:/app/models \
  -p 8000:8000 \
  qwen3-stt
```

**entrypoint.sh 실행 순서**:
1. Redis 서버 시작 (백그라운드)
2. vLLM 모델 서빙 시작 (`:8001`, 백그라운드)
3. vLLM 헬스체크 대기
4. FastAPI 서버 시작 (`:8000`, foreground)

### 8.2 Docker Compose (분리 배포)

```yaml
services:
  redis:
    image: redis:7-alpine
    ports: ["6379:6379"]

  vllm:
    image: qwenllm/qwen3-asr:latest
    deploy:
      resources:
        reservations:
          devices: [{ driver: nvidia, count: 1, capabilities: [gpu] }]

  api:
    build: .
    ports: ["8000:8000"]
    depends_on: [redis, vllm]
```

### 8.3 네트워크 구성 예시

```
                    ┌─ 사내 네트워크 ──────────────────────┐
                    │                                      │
Internet ─── LB ───┤  ┌────────────┐  ┌───────────────┐  │
                    │  │ Backend    │  │ STT Container  │  │
                    │  │ Server     │──│ (GPU)          │  │
                    │  │ :8080      │  │ :8000          │  │
                    │  └────────────┘  └───────────────┘  │
                    │                                      │
                    └──────────────────────────────────────┘

* STT 컨테이너는 내부 네트워크에서만 접근
* Backend가 인증/인가 후 STT API 프록시
* HTTPS는 LB 또는 Backend에서 처리
```

---

## 9. 성능 특성

### 9.1 지연 시간

| 구간 | 지연 | 비고 |
|------|------|------|
| 청크 실시간 STT | 1~3초 | 5초 오디오 청크 기준 |
| STT 재처리 (60초 단위) | 0.5~1초/분 | 배치 처리, 높은 정확도 |
| 화자 분리 | 10~30초/10분 | CPU: 느림, GPU: 빠름 |
| Finalize (정합) | < 1초 | CPU 연산 |

### 9.2 확장성

| 항목 | Dev (0.6B) | Prod (1.7B) |
|------|------------|-------------|
| 동시 세션 | ~4 | ~32 |
| vLLM 배치 | 4 seqs | 32 seqs |
| 초당 청크 처리 | ~4 req/s | ~20 req/s |
| GPU 메모리 | ~4 GB | ~8 GB |

### 9.3 리소스 요구사항

| 리소스 | 최소 | 권장 |
|--------|------|------|
| **GPU** | RTX 3060 (12GB) | A100/B200 MIG |
| **RAM** | 8 GB | 16 GB+ |
| **CPU** | 4 cores | 8 cores+ |
| **Disk** | 50 GB | 100 GB+ (모델 + 오디오) |
| **Redis** | 1 GB | 4 GB |

---

## 10. 보안 고려사항

| 항목 | 현재 상태 | 권장 |
|------|-----------|------|
| 인증 | 없음 | Backend에서 인증 후 프록시 |
| HTTPS | 선택적 (`SSL_KEYFILE`, `SSL_CERTFILE`) | LB에서 TLS 종료 |
| 오디오 저장 | 로컬 디스크 평문 | 암호화 스토리지 또는 사용 후 삭제 정책 |
| 세션 TTL | 4시간 | 비즈니스 요건에 맞게 조정 |
| 네트워크 | 포트 8000 노출 | 내부 네트워크만 접근 허용 |

---

## 11. 모니터링

### 11.1 헬스체크 엔드포인트

| 엔드포인트 | 용도 |
|-----------|------|
| `GET /__ping` | L4 로드밸런서 헬스체크 |
| `GET /api/health` | 서버 상태 + 설정 요약 |
| `GET /v1/selftest/model` | ASR 모델 로딩 확인 |
| `GET /v1/selftest/diarization` | 화자분리 모델 확인 |
| `GET /v1/selftest/pipeline` | 전체 파이프라인 확인 |
| `GET /api/sessions/active` | 활성 세션 수 (부하 모니터링) |

### 11.2 주요 모니터링 지표

| 지표 | 확인 방법 |
|------|-----------|
| 서버 가동시간 | `GET /api/health` → `uptime_sec` |
| 활성 세션 수 | `GET /api/sessions/active` |
| GPU 사용률 | `nvidia-smi` (컨테이너 내부) |
| Redis 메모리 | `redis-cli info memory` |
| 디스크 사용량 | `du -sh data/audio/` |

---

## 12. FAQ

**Q: 최대 몇 시간까지 녹음 가능한가요?**
A: 기본 2시간 (`MAX_SESSION_AUDIO_SEC=7200`). 환경변수로 조정 가능합니다.

**Q: 화자는 최대 몇 명까지 구분되나요?**
A: pyannote 모델 특성상 제한 없으나, 실무적으로 10명 이내에서 정확도가 높습니다.

**Q: 한국어만 지원하나요?**
A: Qwen3-ASR은 다국어를 지원합니다 (한국어, 영어, 중국어, 일본어 등). 언어 자동 감지됩니다.

**Q: 네트워크가 끊기면 어떻게 되나요?**
A: 세션은 TTL(4시간) 동안 유지됩니다. 동일 session_id로 resume 후 이어서 청크를 전송할 수 있습니다. seq 중복은 자동 무시됩니다.

**Q: 실시간 STT 없이 녹음만 하고 나중에 결과를 받을 수 있나요?**
A: `realtime=0`으로 청크를 전송하면 저장만 하고, stop → finalize에서 일괄 처리됩니다.

**Q: 동일 화자가 다른 화자로 분리될 때는?**
A: `DIAR_EMBEDDING_THRESHOLD`를 낮추면 더 관대하게 매칭합니다 (기본 0.55). 반대로 다른 화자가 같은 화자로 합쳐지면 값을 높이세요.

**Q: 겹침 발화(동시 대화)는 어떻게 처리되나요?**
A: pyannote가 감지한 겹침 구간에서 2초 이하의 짧은 발화는 문맥 기반으로 dominant 화자에 흡수하고, 2초 초과의 긴 겹침은 중간점으로 분할하여 각 화자에게 배분합니다.
