# PCM 오디오 처리 & API 통합 가이드

> **대상 독자**: 프론트엔드에서 PCM 오디오를 캡처하여 STT 컨테이너 API를 호출하는 **백엔드/프론트엔드 개발자**

---

## 목차

1. [PCM 오디오 규격](#1-pcm-오디오-규격)
2. [프론트엔드 PCM 캡처 가이드](#2-프론트엔드-pcm-캡처-가이드)
3. [REST API 레퍼런스](#3-rest-api-레퍼런스)
4. [WebSocket API 레퍼런스](#4-websocket-api-레퍼런스)
5. [응답 스키마 상세](#5-응답-스키마-상세)
6. [전체 워크플로우 시퀀스](#6-전체-워크플로우-시퀀스)
7. [에러 처리](#7-에러-처리)
8. [백엔드 통합 예제 (Java/Spring)](#8-백엔드-통합-예제)
9. [시스템 엔드포인트](#9-시스템-엔드포인트)

---

## 1. PCM 오디오 규격

STT 서비스가 요구하는 오디오 포맷:

| 항목 | 값 | 비고 |
|------|-----|------|
| **포맷** | PCM16 (Raw) | WAV 헤더 없이 순수 PCM 바이트 |
| **샘플레이트** | **16,000 Hz** | 필수. 다른 레이트는 다운샘플링 필요 |
| **비트 깊이** | 16-bit (signed, little-endian) | 샘플당 2바이트 |
| **채널** | 모노 (1ch) | 스테레오 전송 시 서버가 자동 믹스다운 |
| **바이트 오더** | Little-endian | JavaScript Int16 기본값과 동일 |
| **청크 크기** | 2~5초 권장 | 1초 미만 → 인식 품질 저하, 10초 초과 → 지연 증가 |
| **최대 청크 크기** | 2 MB | 초과 시 `413` 에러 |

### 바이트 계산

```
1초 오디오 = 16000 samples × 2 bytes = 32,000 bytes (≈31.25 KB)
5초 오디오 = 160,000 bytes (≈156.25 KB)
```

### PCM16 바이트 유효성 검사

서버는 다음을 검증합니다:
- `len(raw) > 0` — 빈 데이터 거부
- `len(raw) % 2 == 0` — PCM16은 짝수 바이트여야 함 (400 에러)
- `len(raw) <= 2MB` — 초과 시 413 에러

---

## 2. 프론트엔드 PCM 캡처 가이드

### 2.1 마이크 캡처 → PCM16 변환

```javascript
// 1. 마이크 접근
const stream = await navigator.mediaDevices.getUserMedia({
  audio: {
    sampleRate: 16000,     // 브라우저가 지원 시
    channelCount: 1,
    echoCancellation: true,
    noiseSuppression: false // 서버에서 노이즈 제거 처리
  }
});

// 2. AudioContext 생성
const audioCtx = new AudioContext({ sampleRate: 16000 });
const source = audioCtx.createMediaStreamSource(stream);

// 3. ScriptProcessor로 PCM 추출 (또는 AudioWorklet 사용)
const BUFFER_SIZE = 4096;
const processor = audioCtx.createScriptProcessor(BUFFER_SIZE, 1, 1);

let pcmBuffer = [];
let chunkDurationSec = 5;
let samplesPerChunk = 16000 * chunkDurationSec;

processor.onaudioprocess = (e) => {
  const float32 = e.inputBuffer.getChannelData(0);
  pcmBuffer.push(...float32);

  if (pcmBuffer.length >= samplesPerChunk) {
    const chunk = pcmBuffer.splice(0, samplesPerChunk);
    sendChunk(float32ToInt16(chunk));
  }
};

source.connect(processor);
processor.connect(audioCtx.destination);
```

### 2.2 Float32 → PCM16(Int16) 변환

```javascript
function float32ToInt16(float32Array) {
  const int16 = new Int16Array(float32Array.length);
  for (let i = 0; i < float32Array.length; i++) {
    const s = Math.max(-1, Math.min(1, float32Array[i]));
    int16[i] = s < 0 ? s * 0x8000 : s * 0x7FFF;
  }
  return int16.buffer; // ArrayBuffer (PCM16 raw bytes)
}
```

### 2.3 다운샘플링 (브라우저 샘플레이트 ≠ 16kHz일 때)

```javascript
function downsample(buffer, fromRate, toRate) {
  if (fromRate === toRate) return buffer;
  const ratio = fromRate / toRate;
  const newLength = Math.round(buffer.length / ratio);
  const result = new Float32Array(newLength);
  for (let i = 0; i < newLength; i++) {
    result[i] = buffer[Math.round(i * ratio)];
  }
  return result;
}

// 사용
const nativeSampleRate = audioCtx.sampleRate; // 예: 48000
const downsampled = downsample(float32Data, nativeSampleRate, 16000);
const pcm16 = float32ToInt16(downsampled);
```

### 2.4 시스템 오디오 캡처 (온라인 미팅용)

```javascript
// 화면 공유 + 시스템 오디오
const displayStream = await navigator.mediaDevices.getDisplayMedia({
  video: true,
  audio: true // 시스템 오디오
});

// 마이크
const micStream = await navigator.mediaDevices.getUserMedia({ audio: true });

// 믹싱
const ctx = new AudioContext({ sampleRate: 16000 });
const dest = ctx.createMediaStreamDestination();
ctx.createMediaStreamSource(displayStream).connect(dest);
ctx.createMediaStreamSource(micStream).connect(dest);

// dest.stream 에서 PCM 추출 (위와 동일 방법)
```

---

## 3. REST API 레퍼런스

**Base URL**: `http://{HOST}:8000`

### 3.1 세션 생성

```
POST /api/session
```

**Request Body**: 없음

**Response** `200`:
```json
{
  "session_id": "a1b2c3d4-...",
  "ttl_sec": 14400
}
```

| 필드 | 타입 | 설명 |
|------|------|------|
| `session_id` | string | 세션 고유 ID (UUID) |
| `ttl_sec` | int | 세션 만료 시간 (초). 기본 4시간 |

---

### 3.2 오디오 청크 업로드

```
POST /api/session/{session_id}/chunk?seq={N}&realtime={0|1}
```

**Path Parameters**:

| 파라미터 | 타입 | 필수 | 설명 |
|----------|------|------|------|
| `session_id` | string | O | 세션 ID |

**Query Parameters**:

| 파라미터 | 타입 | 필수 | 기본값 | 설명 |
|----------|------|------|--------|------|
| `seq` | int | O | - | 청크 순번 (0부터 시작, 중복 전송 시 자동 무시) |
| `realtime` | int | X | `1` | `1`: 실시간 STT 수행, `0`: 저장만 (finalize 때 일괄 처리) |

**Request Body**: `application/octet-stream` — PCM16 raw bytes

**Request Headers**:
```
Content-Type: application/octet-stream
```

**Response** `200`:
```json
{
  "session_id": "a1b2c3d4-...",
  "seq": 0,
  "accepted_seq": 0,
  "chunk_text": "안녕하세요 오늘 회의를 시작하겠습니다",
  "stt_text": "",
  "received_bytes": 160000,
  "start_ms": 0,
  "end_ms": 5000,
  "audio_metrics": {
    "rms_dbfs": -18.5,
    "peak_dbfs": -3.2,
    "clipping_ratio": 0.0,
    "noise_floor_db": -42.1,
    "snr_estimate": 23.6,
    "suggestions": []
  },
  "backlog_hint": "ok",
  "duplicate": false
}
```

| 필드 | 타입 | 설명 |
|------|------|------|
| `session_id` | string | 세션 ID |
| `seq` | int | 요청한 순번 |
| `accepted_seq` | int | 서버가 처리한 순번 |
| `chunk_text` | string | 실시간 STT 결과 텍스트 (`realtime=0`이면 빈 문자열) |
| `received_bytes` | int | 수신한 PCM 바이트 수 |
| `start_ms` | int | 청크 시작 시각 (ms) |
| `end_ms` | int | 청크 종료 시각 (ms) |
| `audio_metrics` | object | 오디오 품질 지표 ([상세](#51-audio_metrics)) |
| `backlog_hint` | string | `"ok"` 또는 `"backpressure"` (처리 지연 시) |
| `duplicate` | bool | 중복 seq 여부 |

**에러 응답**:
- `400` — PCM16 바이트 길이가 홀수
- `413` — 청크 크기 2MB 초과
- `429` — 세션 오디오 시간 한도 초과 또는 세션 중지됨

---

### 3.3 녹음 중단

```
POST /api/session/{session_id}/stop
```

**설명**: 녹음을 중단하고, 백그라운드에서 STT 재처리(60초 단위)와 화자 분리를 시작합니다.

**Response** `200`:
```json
{
  "status": "stopped",
  "ssid": "a1b2c3d4-...",
  "total_audio_sec": 320.5,
  "audio_path": "/app/data/audio/a1b2c3d4-....wav"
}
```

---

### 3.4 처리 상태 조회

```
GET /api/session/{session_id}/status
```

**Response** `200`:
```json
{
  "ssid": "a1b2c3d4-...",
  "status": "processing",
  "stt_done": false,
  "diar_done": false,
  "total_audio_sec": 320.5,
  "audio_path": "/app/data/audio/a1b2c3d4-....wav"
}
```

| 필드 | 타입 | 설명 |
|------|------|------|
| `status` | string | `"recording"`, `"stopped"`, `"processing"`, `"done"` |
| `stt_done` | bool | STT 재처리 완료 여부 |
| `diar_done` | bool | 화자 분리 완료 여부 |

**폴링 권장 간격**: 2~3초

---

### 3.5 최종 결과 조회

```
POST /api/session/{session_id}/finalize
```

**설명**: STT + 화자 분리 결과를 합쳐 타임라인을 생성합니다. `status`에서 `stt_done=true && diar_done=true`일 때 호출하세요.

**Response** `200`:
```json
{
  "ssid": "a1b2c3d4-...",
  "timeline": [
    {
      "speaker": "SPEAKER_00",
      "start": 0.0,
      "end": 3.5,
      "text": "안녕하세요 오늘 회의를 시작하겠습니다"
    },
    {
      "speaker": "SPEAKER_01",
      "start": 3.8,
      "end": 7.2,
      "text": "네 준비되었습니다"
    }
  ],
  "full_text": "안녕하세요 오늘 회의를 시작하겠습니다 네 준비되었습니다",
  "diarization": [
    {"speaker": "SPEAKER_00", "start": 0.0, "end": 3.5},
    {"speaker": "SPEAKER_01", "start": 3.8, "end": 7.2}
  ],
  "total_audio_sec": 320.5,
  "audio_path": "/app/data/audio/a1b2c3d4-....wav"
}
```

| 필드 | 타입 | 설명 |
|------|------|------|
| `timeline` | array | 화자별 발화 타임라인 (시간 순) |
| `timeline[].speaker` | string | 화자 라벨 (`SPEAKER_00`, `SPEAKER_01`, ...) |
| `timeline[].start` | float | 발화 시작 시각 (초) |
| `timeline[].end` | float | 발화 종료 시각 (초) |
| `timeline[].text` | string | 해당 구간 텍스트 |
| `full_text` | string | 전체 텍스트 (화자 구분 없음) |
| `diarization` | array | 화자 분리 원본 구간 |
| `total_audio_sec` | float | 총 오디오 길이 (초) |

---

### 3.6 세션 재개

```
POST /api/session/{session_id}/resume
```

**설명**: 중단된 세션을 재개하여 녹음을 이어갑니다.

**Response** `200`:
```json
{
  "status": "resumed",
  "ssid": "a1b2c3d4-..."
}
```

**에러**: `404` — 세션이 존재하지 않거나 재개 불가

---

### 3.7 부분 결과 조회

```
GET /api/session/{session_id}/partial-results
```

**설명**: finalize 호출 없이 현재까지 누적된 결과를 반환합니다. 녹음 중에도 호출 가능합니다.

---

### 3.8 오디오 파일 다운로드

```
GET /api/session/{session_id}/audio
```

**Response**: `audio/wav` 파일 (녹음된 전체 오디오)

**에러**: `404` — 오디오 파일 미존재 (stop 전에는 WAV 파일 미생성)

---

### 3.9 활성 세션 수

```
GET /api/sessions/active
```

**Response**:
```json
{
  "active_sessions": 5
}
```

---

## 4. WebSocket API 레퍼런스

**Endpoint**: `ws://{HOST}:8000/v1/ws`

REST 대비 장점: 연결 유지 상태에서 양방향 통신, 더 낮은 지연.

### 4.1 메시지 흐름

```
Client                          Server
  │                                │
  │─── start (JSON) ──────────────▶│
  │◀── ack (JSON) ────────────────│
  │                                │
  │─── chunk (Binary) ────────────▶│  ← 반복
  │◀── partial (JSON) ────────────│
  │                                │
  │─── stop (JSON) ───────────────▶│
  │◀── stopped (JSON) ────────────│
  │                                │
  │─── status (JSON) ─────────────▶│  ← 폴링
  │◀── ack (JSON) ────────────────│
  │                                │
  │─── finalize (JSON) ───────────▶│
  │◀── final (JSON) ──────────────│
```

### 4.2 세션 시작

**Client → Server** (JSON text frame):
```json
{
  "type": "start",
  "payload": {
    "ssid": null,
    "sample_rate": 16000,
    "channels": 1,
    "chunk_sec": 5
  }
}
```

| 필드 | 타입 | 필수 | 설명 |
|------|------|------|------|
| `payload.ssid` | string\|null | X | 커스텀 세션 ID. null이면 자동 생성 |
| `payload.sample_rate` | int | O | 샘플레이트 (16000) |
| `payload.channels` | int | X | 채널 수 (기본 1) |
| `payload.chunk_sec` | int | X | 청크 길이 (기본 2초) |

**Server → Client**:
```json
{
  "type": "ack",
  "ssid": "a1b2c3d4-...",
  "accepted_seq": -1
}
```

### 4.3 오디오 청크 전송

**Client → Server** (Binary frame):

바이너리 프레임 구조:
```
[JSON 헤더]\n[Base64 인코딩된 PCM16 바이트]
```

JSON 헤더:
```json
{"ssid": "a1b2c3d4-...", "seq": 0, "t0": null, "realtime": true}
```

| 필드 | 타입 | 필수 | 설명 |
|------|------|------|------|
| `ssid` | string | O | 세션 ID |
| `seq` | int | O | 청크 순번 (0부터) |
| `t0` | float\|null | X | 청크 시작 시각 (커스텀 타임스탬프) |
| `realtime` | bool | X | 실시간 STT 여부 (기본 true) |

**JavaScript 전송 예제**:
```javascript
function sendChunkWS(ws, ssid, seq, pcm16ArrayBuffer) {
  const header = JSON.stringify({ ssid, seq, t0: null, realtime: true });
  const headerBytes = new TextEncoder().encode(header + "\n");
  const base64 = btoa(String.fromCharCode(...new Uint8Array(pcm16ArrayBuffer)));
  const base64Bytes = new TextEncoder().encode(base64);

  const frame = new Uint8Array(headerBytes.length + base64Bytes.length);
  frame.set(headerBytes, 0);
  frame.set(base64Bytes, headerBytes.length);

  ws.send(frame.buffer);
}
```

**Server → Client**:
```json
{
  "type": "partial",
  "ssid": "a1b2c3d4-...",
  "seq": 0,
  "partial_text": "안녕하세요",
  "audio_metrics": {
    "rms_dbfs": -18.5,
    "peak_dbfs": -3.2,
    "clipping_ratio": 0.0,
    "noise_floor_db": -42.1,
    "snr_estimate": 23.6,
    "suggestions": []
  },
  "backlog_hint": "ok"
}
```

### 4.4 세션 중단

```json
{"type": "stop", "payload": {"ssid": "a1b2c3d4-..."}}
```

### 4.5 상태 확인

```json
{"type": "status", "payload": {"ssid": "a1b2c3d4-..."}}
```

### 4.6 최종 결과 요청

```json
{"type": "finalize", "payload": {"ssid": "a1b2c3d4-..."}}
```

**Server → Client**: `{"type": "final", "timeline": [...], "full_text": "...", ...}`

---

## 5. 응답 스키마 상세

### 5.1 audio_metrics

오디오 품질 지표. 프론트엔드에서 사용자에게 피드백을 줄 때 활용합니다.

```json
{
  "rms_dbfs": -18.5,
  "peak_dbfs": -3.2,
  "clipping_ratio": 0.0,
  "noise_floor_db": -42.1,
  "snr_estimate": 23.6,
  "suggestions": []
}
```

| 필드 | 타입 | 설명 |
|------|------|------|
| `rms_dbfs` | float | RMS 음량 (dBFS). -35 미만이면 너무 조용함 |
| `peak_dbfs` | float | 피크 음량 (dBFS) |
| `clipping_ratio` | float | 클리핑 비율 (0.01 초과 시 클리핑 위험) |
| `noise_floor_db` | float | 노이즈 플로어 (dB) |
| `snr_estimate` | float | SNR 추정값 (dB). 8 미만이면 잡음 심함 |
| `suggestions` | string[] | 서버 제안 목록 |

### suggestions 값:

| 값 | 의미 | 프론트엔드 대응 |
|----|------|----------------|
| `INPUT_TOO_QUIET` | 입력 음량이 너무 낮음 (RMS < -35 dBFS) | "마이크를 가까이 하거나 볼륨을 높여주세요" |
| `CLIPPING_RISK` | 클리핑 발생 가능 (clipping_ratio > 1%) | "마이크 볼륨을 줄여주세요" |
| `HIGH_NOISE` | 잡음이 심함 (SNR < 8 dB) | "조용한 환경에서 녹음해주세요" |

### 5.2 backlog_hint

| 값 | 의미 | 대응 |
|----|------|------|
| `"ok"` | 서버 정상 처리 중 | 정상 진행 |
| `"backpressure"` | 서버 처리 지연 | 청크 전송 간격을 늘리거나 `realtime=0` 전환 고려 |

### 5.3 timeline 항목

```json
{
  "speaker": "SPEAKER_00",
  "start": 0.0,
  "end": 3.5,
  "text": "안녕하세요 오늘 회의를 시작하겠습니다"
}
```

- `speaker`: `SPEAKER_00`, `SPEAKER_01`, ... 형식. 최대 감지 가능 화자 수는 모델에 따라 다름
- `start`/`end`: 초 단위 시각
- `text`: 해당 시간 구간의 인식 텍스트

---

## 6. 전체 워크플로우 시퀀스

### REST API 워크플로우

```
┌────────────┐        ┌────────────┐        ┌─────────────┐
│  Frontend   │        │  Backend   │        │ STT Container│
│  (Browser)  │        │  (Spring)  │        │  (:8000)     │
└─────┬──────┘        └─────┬──────┘        └──────┬──────┘
      │                      │                      │
      │  1. 녹음 시작 요청    │                      │
      │─────────────────────▶│                      │
      │                      │  POST /api/session   │
      │                      │─────────────────────▶│
      │                      │◀─ {session_id, ttl} ─│
      │◀── session_id ───────│                      │
      │                      │                      │
      │  2. PCM 청크 전송     │                      │
      │  (5초마다)           │                      │
      │── pcm16 bytes ──────▶│                      │
      │                      │  POST /chunk?seq=N   │
      │                      │  Body: pcm16 raw     │
      │                      │─────────────────────▶│
      │                      │◀── {chunk_text, ...} │
      │◀── chunk_text ───────│                      │
      │                      │                      │
      │  3. 녹음 종료         │                      │
      │─────────────────────▶│                      │
      │                      │  POST /stop          │
      │                      │─────────────────────▶│
      │                      │◀── {status: stopped} │
      │                      │                      │
      │  4. 상태 폴링         │                      │
      │─────────────────────▶│  GET /status         │
      │                      │─────────────────────▶│
      │                      │◀── {stt_done, ...}   │
      │◀── "처리중..." ──────│                      │
      │                      │                      │
      │  5. 결과 요청         │                      │
      │─────────────────────▶│  POST /finalize      │
      │                      │─────────────────────▶│
      │                      │◀── {timeline, ...}   │
      │◀── 최종 결과 ─────────│                      │
```

### 단계별 상세

| 단계 | API | 설명 |
|------|-----|------|
| **① 세션 생성** | `POST /api/session` | 세션 ID 발급. 프론트에서 localStorage에 저장 권장 |
| **② 청크 업로드** | `POST /api/session/{id}/chunk?seq=N&realtime=1` | PCM16 바이트를 Body에 직접 전송. `seq`는 0부터 순차 증가 |
| **③ 녹음 중단** | `POST /api/session/{id}/stop` | 백그라운드 STT 재처리 + 화자 분리 시작 |
| **④ 상태 폴링** | `GET /api/session/{id}/status` | `stt_done && diar_done`이 모두 true가 될 때까지 폴링 |
| **⑤ 결과 조회** | `POST /api/session/{id}/finalize` | 화자별 타임라인, 전체 텍스트 반환 |

---

## 7. 에러 처리

### HTTP 상태 코드

| 코드 | 원인 | 대응 |
|------|------|------|
| `400` | PCM16 바이트 길이 홀수 | 프론트엔드 PCM 변환 로직 확인 |
| `404` | 세션 없음 / 오디오 미존재 | 세션 생성 후 사용, stop 후 audio 접근 |
| `413` | 청크 2MB 초과 | 청크 크기 줄이기 (5초 이하) |
| `429` | 오디오 시간 한도 초과 (2시간) 또는 세션 중지됨 | 새 세션 생성 |

### seq 중복 처리

- 같은 `seq` 번호로 재전송하면 서버가 자동으로 무시합니다 (`duplicate: true`)
- 네트워크 불안정 시 동일 청크를 재전송해도 안전합니다

### 세션 복구

- `ttl_sec` (기본 4시간) 내에는 세션이 유지됩니다
- 페이지 새로고침 후 `GET /api/session/{id}/status`로 세션 존재 확인
- `POST /api/session/{id}/resume`으로 녹음 재개 가능

---

## 8. 백엔드 통합 예제

### 8.1 Java/Spring Boot — 청크 업로드 프록시

```java
@RestController
@RequestMapping("/meeting")
public class MeetingController {

    private final WebClient sttClient = WebClient.builder()
        .baseUrl("http://stt-container:8000")
        .build();

    // 세션 생성
    @PostMapping("/start")
    public Mono<Map> startSession() {
        return sttClient.post()
            .uri("/api/session")
            .retrieve()
            .bodyToMono(Map.class);
    }

    // PCM 청크 업로드 (프론트에서 받은 바이너리 그대로 전달)
    @PostMapping("/chunk/{sessionId}")
    public Mono<Map> uploadChunk(
        @PathVariable String sessionId,
        @RequestParam int seq,
        @RequestBody byte[] pcmData
    ) {
        return sttClient.post()
            .uri(uriBuilder -> uriBuilder
                .path("/api/session/{id}/chunk")
                .queryParam("seq", seq)
                .queryParam("realtime", 1)
                .build(sessionId))
            .contentType(MediaType.APPLICATION_OCTET_STREAM)
            .bodyValue(pcmData)
            .retrieve()
            .bodyToMono(Map.class);
    }

    // 녹음 중단
    @PostMapping("/stop/{sessionId}")
    public Mono<Map> stop(@PathVariable String sessionId) {
        return sttClient.post()
            .uri("/api/session/{id}/stop", sessionId)
            .retrieve()
            .bodyToMono(Map.class);
    }

    // 상태 조회
    @GetMapping("/status/{sessionId}")
    public Mono<Map> status(@PathVariable String sessionId) {
        return sttClient.get()
            .uri("/api/session/{id}/status", sessionId)
            .retrieve()
            .bodyToMono(Map.class);
    }

    // 최종 결과
    @PostMapping("/finalize/{sessionId}")
    public Mono<Map> finalize(@PathVariable String sessionId) {
        return sttClient.post()
            .uri("/api/session/{id}/finalize", sessionId)
            .retrieve()
            .bodyToMono(Map.class);
    }
}
```

### 8.2 Python (requests) — 간단 테스트

```python
import requests

BASE = "http://localhost:8000"

# 1. 세션 생성
resp = requests.post(f"{BASE}/api/session")
session_id = resp.json()["session_id"]

# 2. PCM 파일 읽어서 청크 전송
with open("recording.pcm", "rb") as f:
    seq = 0
    while True:
        chunk = f.read(160000)  # 5초분
        if not chunk:
            break
        resp = requests.post(
            f"{BASE}/api/session/{session_id}/chunk",
            params={"seq": seq, "realtime": 1},
            data=chunk,
            headers={"Content-Type": "application/octet-stream"},
        )
        print(f"seq={seq}: {resp.json().get('chunk_text', '')}")
        seq += 1

# 3. 중단
requests.post(f"{BASE}/api/session/{session_id}/stop")

# 4. 폴링
import time
while True:
    st = requests.get(f"{BASE}/api/session/{session_id}/status").json()
    if st.get("stt_done") and st.get("diar_done"):
        break
    time.sleep(2)

# 5. 결과
result = requests.post(f"{BASE}/api/session/{session_id}/finalize").json()
for turn in result["timeline"]:
    print(f"[{turn['speaker']}] {turn['start']:.1f}s~{turn['end']:.1f}s: {turn['text']}")
```

### 8.3 cURL

```bash
# 세션 생성
SESSION_ID=$(curl -s -X POST http://localhost:8000/api/session | jq -r '.session_id')

# 청크 전송 (PCM 파일)
curl -X POST "http://localhost:8000/api/session/${SESSION_ID}/chunk?seq=0&realtime=1" \
  -H "Content-Type: application/octet-stream" \
  --data-binary @chunk_0.pcm

# 중단
curl -X POST "http://localhost:8000/api/session/${SESSION_ID}/stop"

# 상태 확인
curl "http://localhost:8000/api/session/${SESSION_ID}/status"

# 최종 결과
curl -X POST "http://localhost:8000/api/session/${SESSION_ID}/finalize"

# 오디오 다운로드
curl -o meeting.wav "http://localhost:8000/api/session/${SESSION_ID}/audio"
```

---

## 9. 시스템 엔드포인트

운영/모니터링용 엔드포인트입니다.

| 메서드 | 경로 | 설명 |
|--------|------|------|
| `GET` | `/__ping` | 헬스체크 (`"OK"` 반환) |
| `GET` | `/api/health` | 서버 가동시간, 설정 요약 |
| `GET` | `/api/runtime` | 런타임 설정 전체 (모델명, 전처리 설정 등) |
| `GET` | `/v1/selftest/model` | ASR 모델 로딩 상태 |
| `GET` | `/v1/selftest/diarization` | 화자 분리 모델 로딩 상태 |
| `GET` | `/v1/selftest/pipeline` | 전체 파이프라인 상태 |

### 헬스체크 응답 예시

```json
{
  "status": "ok",
  "uptime_sec": 3600.5,
  "config": {
    "stt_final_chunk_sec": 60,
    "max_session_audio_sec": 7200,
    "session_ttl_sec": 14400,
    "vllm_model": "Qwen/Qwen3-ASR-0.6B"
  }
}
```

---

## 부록: v1 API (고급)

프론트엔드 전용 `/api/` 라우트 외에, 더 세밀한 제어가 가능한 `/v1/` 라우트도 있습니다.

| 메서드 | 경로 | 설명 |
|--------|------|------|
| `POST` | `/v1/sessions/` | 세션 생성 (sample_rate, channels, chunk_sec 직접 지정) |
| `POST` | `/v1/sessions/{ssid}/chunk?seq=N&t0=&realtime=1` | 청크 업로드 (multipart file) |
| `POST` | `/v1/sessions/{ssid}/stop` | 녹음 중단 |
| `GET` | `/v1/sessions/{ssid}/status` | 상태 조회 |
| `POST` | `/v1/sessions/{ssid}/finalize` | 결과 조회 |

### v1 세션 생성 요청 Body

```json
{
  "ssid": "custom-session-id-or-null",
  "sample_rate": 16000,
  "channels": 1,
  "chunk_sec": 5
}
```

### v1 청크 업로드

`Content-Type: multipart/form-data` — `file` 필드에 PCM16 바이너리 첨부
