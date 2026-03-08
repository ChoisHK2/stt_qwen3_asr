# FAQ

## Q1. 왜 WS가 기본이고 REST 폴백이 있나요?
A. 2~5초 chunk 스트리밍에서 저지연 partial 반응이 필요해 WS가 기본입니다. 다만 방화벽/프록시 이슈를 고려해 동일 기능의 REST 폴백을 제공합니다.

## Q2. ssid/seq/t0는 각각 무엇인가요?
A. `ssid`는 세션 식별자, `seq`는 chunk 순번(idempotency 키), `t0`는 클라이언트 기준 chunk 시작 시각(옵션)입니다. 재연결 시 `status.last_accepted_seq` 이후만 재전송하면 복구됩니다.

## Q3. partial은 무엇이며 언제 갱신되나요?
A. chunk ingest 직후 전처리 + 저지연 ASR 결과를 반환하는 중간 자막입니다. `PARTIAL_MODE=on`일 때 chunk마다 `partial` 이벤트/응답이 나갑니다.

## Q4. 왜 별도 큐/워커가 없나요?
A. vLLM은 자체 continuous batching으로 동시 요청을 큐잉합니다. 별도 Redis 큐를 두면 오히려 불필요한 복잡성과 지연이 추가됩니다. API에서 직접 vLLM HTTP API를 호출하고, `MAX_CONCURRENT_ASR` 세마포어로 동시 요청 수만 제한합니다.

## Q5. 전처리(증폭/노이즈제거/VAD) 효과는?
A. 저음량/잡음 환경에서 partial 품질 안정화에 유리합니다. 기본은 downmix, DC 제거, RMS 타깃 증폭+리미터, 노이즈 제거가 ON이며 VAD는 옵션입니다.

## Q6. diarization 결과가 왜 잘게 나뉘나요?
A. pyannote는 짧은 발화 전환/겹말을 민감하게 분리합니다. 뭉침이 과하면 `MERGE_MODE`, `MERGE_GAP_SEC`, `MIN_TURN_SEC`, `MIN_WORDS_PER_TURN`를 조절하세요.

## Q7. finalize가 오래 걸리면?
A. 녹음 중지(stop) 시 백그라운드로 STT 재처리와 화자분리가 시작됩니다. 클라이언트는 `/status` 폴링으로 완료를 확인한 후 `/finalize`를 호출합니다.

## Q8. 0.6B와 1.7B 모델 전환은?
A. `.env.dev`(0.6B) 또는 `.env.prod`(1.7B)를 `.env`에 복사하고, docker compose에서 해당 profile(`--profile dev` / `--profile prod`)을 사용하세요.

## Q9. 100 커넥션을 어떻게 지원하나요?
A. vLLM의 `--max-num-seqs 32`로 동시 추론을 처리하고, 나머지 요청은 vLLM이 자체 큐잉합니다. 5초 청크 × 100 커넥션 = 초당 ~20 요청이므로 B200 MIG 30GB에서 충분합니다.

## Q10. 흔한 오류와 해결법은?
A.
- 모델 경로 불일치: `MODEL_DIR`/vLLM `--model` 경로 확인
- Redis 연결 실패: `REDIS_URL` 및 compose network 확인
- pyannote 로드 실패: 토큰/모델 캐시 확인
- MIG OOM: vLLM `--max-num-seqs`, `--gpu-memory-utilization` 하향

## Q11. 성능 튜닝 포인트는?
A. `MAX_CONCURRENT_ASR`, chunk 2s vs 5s, noise mode FAST/QUALITY, vLLM `--max-num-seqs`를 함께 조정하세요. 지연 최적화는 2s+낮은 동시성, 처리량 최적화는 5s+큐 확장이 유리합니다.

## Q12. 오프라인 배포 절차 핵심은?
A. 온라인에서 모델 스냅샷(`/models`) + Docker 이미지 tar를 만든 뒤 오프라인 서버로 이동해 `docker load` 후 로컬 모델 경로로 실행합니다. API 이미지는 공통으로 유지하고, vLLM 런타임 이미지만 GPU 스택별로 분리하세요.
