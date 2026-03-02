# FAQ

## Q1. 왜 WS가 기본이고 REST 폴백이 있나요?
A. 2~4초 chunk 스트리밍에서 저지연 partial 반응이 필요해 WS가 기본입니다. 다만 방화벽/프록시 이슈를 고려해 동일 기능의 REST 폴백을 제공합니다.

## Q2. ssid/seq/t0는 각각 무엇인가요?
A. `ssid`는 세션 식별자, `seq`는 chunk 순번(idempotency 키), `t0`는 클라이언트 기준 chunk 시작 시각(옵션)입니다. 재연결 시 `status.last_accepted_seq` 이후만 재전송하면 복구됩니다.

## Q3. partial은 무엇이며 언제 갱신되나요?
A. chunk ingest 직후 전처리 + 저지연 ASR 결과를 반환하는 중간 자막입니다. `PARTIAL_MODE=on`일 때 chunk마다 `partial` 이벤트/응답이 나갑니다.

## Q4. backlog_hint는 어떻게 해석하나요?
A. `ok`: 정상 전송, `slow_down`: 업로드 간격 완화 권장, `paused`: 일시 중지 후 status 폴링 권장입니다. `paused`에서 계속 밀어넣으면 429/WS error가 날 수 있습니다.

## Q5. 전처리(증폭/노이즈제거/VAD) 효과는?
A. 저음량/잡음 환경에서 partial 품질 안정화에 유리합니다. 기본은 downmix, DC 제거, RMS 타깃 증폭+리미터, 노이즈 제거가 ON이며 VAD는 옵션입니다.

## Q6. diarization 결과가 왜 잘게 나뉘나요?
A. pyannote는 짧은 발화 전환/겹말을 민감하게 분리합니다. 뭉침이 과하면 `MERGE_MODE`, `MERGE_GAP_SEC`, `MIN_TURN_SEC`, `MIN_WORDS_PER_TURN`를 조절하세요.

## Q7. finalize가 오래 걸리면?
A. 긴 세션은 비동기(202 + job_id) 패턴으로 확장할 수 있게 설계했습니다. 현재 레퍼런스는 동기 반환 중심이며, 임계치(`FINALIZE_ASYNC_THRESHOLD_SEC`) 기반 분기를 추가하기 쉽습니다.

## Q8. AMD PC(CUDA 없음)에서도 selftest가 되나요?
A. 가능합니다. 실추론 대신 import/로드/무결성 체크로 graceful degrade 하며, GPU 서버 배포 전 구성 오류를 사전 검출하는 용도입니다.

## Q9. 오프라인 배포 절차 핵심은?
A. 온라인에서 모델 스냅샷(`/models`) + Docker 이미지 tar를 만든 뒤 오프라인 서버로 이동해 `docker load` 후 로컬 모델 경로로 실행합니다.

## Q10. 흔한 오류와 해결법은?
A.
- 모델 경로 불일치: `MODEL_DIR`/vLLM `--model` 경로 확인
- Redis 연결 실패: `REDIS_URL` 및 compose network 확인
- pyannote 로드 실패: 토큰/모델 캐시 확인
- MIG OOM: `WORKER_CONCURRENCY`, vLLM `--max-num-seqs` 하향

## Q11. 성능 튜닝 포인트는?
A. `WORKER_CONCURRENCY`, `GLOBAL_QUEUE_LIMIT`, chunk 2s vs 4s, noise mode FAST/QUALITY, vLLM `--max-num-seqs`를 함께 조정하세요. 지연 최적화는 2s+낮은 동시성, 처리량 최적화는 4s+큐 확장이 유리합니다.

## Q12. AMD(ROCm)에서 테스트하고, 배포는 NVIDIA(CUDA)로 해도 되나요?
A. 네, 권장되는 접근입니다. API/worker 이미지는 공통으로 유지하고, vLLM 런타임 이미지만 GPU 스택별(ROCm/CUDA)로 분리하세요.
- 로컬(ROCm): 기능 검증(WS/REST, 세션 복구, 백프레셔, selftest)
- 오프라인 서버(CUDA): 실성능/실운영 검증
주의: 성능 지표(지연/처리량)는 ROCm과 CUDA가 다르므로 최종 튜닝은 반드시 CUDA(MIG)에서 재측정해야 합니다.

