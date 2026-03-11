from __future__ import annotations

import logging
import os
import wave
from collections import Counter, defaultdict

import numpy as np

from core.models import DiarTurn

logger = logging.getLogger("qwen3-asr.diarization")

# 싱글턴 파이프라인 (기존 diar.py 패턴: 한 번 로드 후 재사용)
_pipeline = None

# 20분 단위 청킹 (오버랩 2분)
DIAR_CHUNK_SEC = 20 * 60
DIAR_OVERLAP_SEC = 2 * 60


def _resolve_diar_device(requested: str) -> str:
    """auto → cuda(있으면) / cpu(없으면)."""
    req = (requested or "auto").strip().lower()
    if req and req != "auto":
        return req
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


def get_pipeline(model_path: str, token: str | None = None, device: str = "cpu"):
    """pyannote 파이프라인을 싱글턴으로 로드한다."""
    global _pipeline
    if _pipeline is not None:
        return _pipeline

    from pyannote.audio import Pipeline
    import torch

    config_path = os.path.join(model_path, "config.yaml")

    if os.path.isfile(config_path):
        logger.info("Loading pyannote pipeline from local: %s", model_path)
        _pipeline = Pipeline.from_pretrained(model_path)
    elif token:
        logger.info("Loading pyannote pipeline from HuggingFace with token")
        _pipeline = Pipeline.from_pretrained(model_path, token=token)
    else:
        raise RuntimeError(
            f"config.yaml not found at {config_path} and no PYANNOTE_TOKEN set. "
            f"Run scripts/download_models.sh to download pyannote models."
        )

    _pipeline.to(torch.device(device))
    return _pipeline


def _diarize_waveform(waveform, sample_rate: int) -> list[DiarTurn]:
    """단일 waveform tensor를 diarize한다."""
    pipeline = _pipeline
    if not pipeline:
        raise RuntimeError("Diarization pipeline not loaded. Call load() first.")

    result = pipeline({"waveform": waveform, "sample_rate": sample_rate})
    annotation = getattr(result, "speaker_diarization", result)

    turns: list[DiarTurn] = []
    for seg, _, speaker in annotation.itertracks(yield_label=True):
        turns.append(DiarTurn(speaker=str(speaker), start=float(seg.start), end=float(seg.end)))
    return turns


def _map_speakers_across_chunks(
    chunk_results: list[tuple[float, list[DiarTurn]]],
) -> list[DiarTurn]:
    """여러 청크의 diar 결과를 하나로 합치면서 화자 ID를 통일한다.

    전략: 오버랩 구간에서 화자 매핑을 결정한다.
    - 청크 N과 청크 N+1의 오버랩 구간을 비교
    - 오버랩 구간에서 시간 겹침이 가장 큰 화자 쌍을 매핑
    - 매핑되지 않은 화자는 새 글로벌 ID 부여
    """
    if len(chunk_results) <= 1:
        if not chunk_results:
            return []
        offset, turns = chunk_results[0]
        return [DiarTurn(speaker=t.speaker, start=t.start + offset, end=t.end + offset) for t in turns]

    # 글로벌 화자 ID 카운터
    global_id_counter = 0
    # chunk_idx → {local_speaker: global_speaker}
    speaker_maps: list[dict[str, str]] = []

    # 첫 번째 청크: 로컬 화자 → 글로벌 화자 직접 할당
    first_offset, first_turns = chunk_results[0]
    first_map: dict[str, str] = {}
    for t in first_turns:
        if t.speaker not in first_map:
            first_map[t.speaker] = f"SPEAKER_{global_id_counter:02d}"
            global_id_counter += 1
    speaker_maps.append(first_map)

    # 이후 청크: 오버랩 구간으로 화자 매핑
    for ci in range(1, len(chunk_results)):
        curr_offset, curr_turns = chunk_results[ci]
        prev_offset, prev_turns = chunk_results[ci - 1]
        prev_map = speaker_maps[ci - 1]

        # 오버랩 구간: curr 시작 ~ prev 끝 (글로벌 시간 기준)
        overlap_start = curr_offset
        overlap_end = prev_offset + DIAR_CHUNK_SEC  # prev 청크의 끝

        # 오버랩 구간에서 prev/curr 화자별 시간 점유 계산
        # prev turns (글로벌 시간)
        prev_global = [
            DiarTurn(speaker=prev_map.get(t.speaker, t.speaker),
                     start=t.start + prev_offset, end=t.end + prev_offset)
            for t in prev_turns
        ]
        # curr turns (글로벌 시간)
        curr_global = [
            DiarTurn(speaker=t.speaker, start=t.start + curr_offset, end=t.end + curr_offset)
            for t in curr_turns
        ]

        # 오버랩 구간 내 겹침 행렬 계산
        overlap_matrix: dict[tuple[str, str], float] = defaultdict(float)
        for pt in prev_global:
            for ct in curr_global:
                ov = max(0.0, min(pt.end, ct.end, overlap_end) - max(pt.start, ct.start, overlap_start))
                if ov > 0:
                    overlap_matrix[(ct.speaker, pt.speaker)] += ov

        # 탐욕적 매핑: 겹침이 큰 순서대로 1:1 매핑
        curr_map: dict[str, str] = {}
        used_globals: set[str] = set()
        for (local, glob), ov in sorted(overlap_matrix.items(), key=lambda x: -x[1]):
            if local not in curr_map and glob not in used_globals:
                curr_map[local] = glob
                used_globals.add(glob)

        # 소거법: 오버랩에 안 나온 로컬 화자 ↔ prev에 있었지만 아직 매핑 안 된 글로벌 화자
        all_prev_globals = set(prev_map.values())
        unmapped_locals = [t.speaker for t in curr_turns if t.speaker not in curr_map]
        unmapped_globals = all_prev_globals - used_globals
        # 발화 시간 순서로 1:1 매핑 (확실하지 않으면 새 ID가 안전하지만, 대부분 동일 화자)
        unmapped_locals_unique = list(dict.fromkeys(unmapped_locals))
        unmapped_globals_sorted = sorted(unmapped_globals)
        for local, glob in zip(unmapped_locals_unique, unmapped_globals_sorted):
            curr_map[local] = glob
            used_globals.add(glob)
            logger.info("Diar chunk %d: elimination mapping %s → %s", ci, local, glob)

        # 그래도 남은 로컬 화자 → 새 글로벌 ID
        for t in curr_turns:
            if t.speaker not in curr_map:
                curr_map[t.speaker] = f"SPEAKER_{global_id_counter:02d}"
                global_id_counter += 1

        speaker_maps.append(curr_map)

        logger.info(
            "Diar chunk %d→%d speaker mapping (overlap %.1f-%.1fs): %s",
            ci - 1, ci, overlap_start, overlap_end, curr_map,
        )

    # 모든 청크의 turn을 글로벌 시간 + 글로벌 화자로 변환
    all_turns: list[DiarTurn] = []
    for ci, (offset, turns) in enumerate(chunk_results):
        smap = speaker_maps[ci]
        for t in turns:
            global_start = t.start + offset
            global_end = t.end + offset
            global_speaker = smap.get(t.speaker, t.speaker)

            # 오버랩 구간 중복 제거: 이전 청크와 겹치는 부분은 이전 청크가 담당
            if ci > 0:
                prev_offset = chunk_results[ci - 1][0]
                prev_chunk_end = prev_offset + DIAR_CHUNK_SEC
                if global_start < prev_chunk_end:
                    # 이 turn이 오버랩 구간에 걸침 → 오버랩 이후 부분만 사용
                    global_start = max(global_start, prev_chunk_end - DIAR_OVERLAP_SEC / 2)
                    if global_start >= global_end:
                        continue

            all_turns.append(DiarTurn(speaker=global_speaker, start=global_start, end=global_end))

    return sorted(all_turns, key=lambda t: t.start)


class DiarizationClient:
    """화자분리 클라이언트 (pyannote.audio).

    싱글턴 파이프라인 + 동기 diarize 메서드.
    20분 초과 오디오는 자동 청킹 + 크로스 청크 화자 매핑.
    """

    def __init__(self):
        self._loaded = False

    def load(self, model_path: str, token: str | None = None, device: str = "cpu"):
        if self._loaded:
            return
        get_pipeline(model_path, token, device)
        self._loaded = True

    def diarize(self, wav_path: str) -> list[DiarTurn]:
        import torch

        with wave.open(wav_path, "rb") as wf:
            n_channels = wf.getnchannels()
            sample_rate = wf.getframerate()
            pcm = wf.readframes(wf.getnframes())

        audio = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
        if n_channels > 1:
            audio = audio.reshape(-1, n_channels).mean(axis=1)

        total_sec = len(audio) / sample_rate

        # 20분 이하: 기존 방식 (단일 처리)
        if total_sec <= DIAR_CHUNK_SEC + 60:
            logger.info("Diarize single pass: %.1f sec", total_sec)
            waveform = torch.from_numpy(audio).unsqueeze(0)
            return _diarize_waveform(waveform, sample_rate)

        # 20분 초과: 청크 단위 처리
        step_samples = (DIAR_CHUNK_SEC - DIAR_OVERLAP_SEC) * sample_rate
        chunk_samples = DIAR_CHUNK_SEC * sample_rate
        chunk_results: list[tuple[float, list[DiarTurn]]] = []

        offset_samples = 0
        chunk_idx = 0
        while offset_samples < len(audio):
            end_samples = min(offset_samples + chunk_samples, len(audio))
            chunk_audio = audio[offset_samples:end_samples]
            chunk_sec = len(chunk_audio) / sample_rate
            offset_sec = offset_samples / sample_rate

            logger.info(
                "Diarize chunk %d: offset=%.1fs, duration=%.1fs",
                chunk_idx, offset_sec, chunk_sec,
            )

            waveform = torch.from_numpy(chunk_audio).unsqueeze(0)
            turns = _diarize_waveform(waveform, sample_rate)
            chunk_results.append((offset_sec, turns))

            offset_samples += step_samples
            chunk_idx += 1

        logger.info("Diarized %d chunks, mapping speakers across chunks...", len(chunk_results))
        return _map_speakers_across_chunks(chunk_results)
