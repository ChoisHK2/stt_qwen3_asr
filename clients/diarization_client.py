from __future__ import annotations

import logging
import os
import warnings
import wave
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np

from core.models import DiarTurn

# ── pyannote 관련 불필요한 경고 억제 ──────────────────────────────
# torchcodec 미설치 경고 (waveform dict 방식 사용 시 정상 작동)
warnings.filterwarnings("ignore", message=".*torchcodec is not installed correctly.*")
# TF32 비활성화 경고 (pyannote 재현성 위해 자동 비활성화, 정상 동작)
warnings.filterwarnings("ignore", message=".*TensorFloat-32.*TF32.*")
# std() degrees of freedom 경고 (짧은 오디오에서 발생, 무해)
warnings.filterwarnings("ignore", message=".*std\\(\\): degrees of freedom is <= 0.*")
# return_embeddings 미지원 경고 (community 모델, fallback 처리됨)
warnings.filterwarnings("ignore", message=".*Ignoring unexpected keyword arguments: return_embeddings.*")

logger = logging.getLogger("qwen3-asr.diarization")

# 싱글턴 파이프라인 / 임베딩 모델
_pipeline = None
_embedding_inference = None

# 20분 단위 청킹 (오버랩 2분) — 단일 diarize() 호출 시 사용
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


def get_embedding_inference(device: str = "cpu"):
    """pyannote 임베딩 모델을 싱글턴으로 로드한다.

    diarization 파이프라인 내부의 임베딩 모델을 재사용한다.
    파이프라인이 없으면 None을 반환한다.

    community 모델(speaker-diarization-community-1) 등 일부 파이프라인에서는
    embedding 속성이 모델 인스턴스가 아닌 dict/str(설정)로 반환되므로,
    이 경우 별도로 모델을 로드한다.
    """
    global _embedding_inference
    if _embedding_inference is not None:
        return _embedding_inference

    if _pipeline is None:
        logger.warning("Cannot load embedding inference: pipeline not loaded yet")
        return None

    try:
        from pyannote.audio import Inference, Model
        import torch

        embedding_model = _pipeline.embedding

        # community 모델 등에서는 embedding이 dict 또는 str(설정값)로 반환됨
        if isinstance(embedding_model, dict):
            # dict 내부에서 모델 이름 추출 시도
            model_id = embedding_model.get("embedding", None)
            if model_id and isinstance(model_id, str):
                logger.info("Loading embedding model from config: %s", model_id)
                embedding_model = Model.from_pretrained(model_id)
            else:
                logger.warning(
                    "Pipeline embedding is a dict without loadable model reference, "
                    "embedding-based speaker matching disabled"
                )
                return None
        elif isinstance(embedding_model, str):
            logger.info("Loading embedding model from name: %s", embedding_model)
            embedding_model = Model.from_pretrained(embedding_model)
        elif not hasattr(embedding_model, "to"):
            logger.warning(
                "Pipeline embedding is not a model (type=%s), "
                "embedding-based speaker matching disabled",
                type(embedding_model).__name__,
            )
            return None

        _embedding_inference = Inference(embedding_model, window="whole")
        _embedding_inference.to(torch.device(device))
        logger.info("Embedding inference loaded successfully")
        return _embedding_inference
    except Exception as e:
        logger.warning("Failed to load embedding inference: %s", e)
        return None


def _diarize_waveform(
    waveform, sample_rate: int, return_embeddings: bool = False,
) -> list[DiarTurn] | tuple[list[DiarTurn], dict[str, np.ndarray]]:
    """단일 waveform tensor를 diarize한다.

    return_embeddings=True이면 (turns, {speaker: centroid_embedding}) 튜플을 반환한다.
    pyannote 3.0+의 return_embeddings 파라미터를 활용하여
    파이프라인 한 번 실행으로 centroid 임베딩까지 추출한다.
    """
    pipeline = _pipeline
    if not pipeline:
        raise RuntimeError("Diarization pipeline not loaded. Call load() first.")

    audio_input = {"waveform": waveform, "sample_rate": sample_rate}

    if return_embeddings:
        try:
            result = pipeline(audio_input, return_embeddings=True)
            # return_embeddings=True → (Annotation, np.ndarray) 튜플
            if isinstance(result, tuple) and len(result) == 2:
                annotation, centroids = result
                annotation = getattr(annotation, "speaker_diarization", annotation)
                labels = list(annotation.labels())

                turns: list[DiarTurn] = []
                for seg, _, speaker in annotation.itertracks(yield_label=True):
                    turns.append(DiarTurn(speaker=str(speaker), start=float(seg.start), end=float(seg.end)))

                embeddings: dict[str, np.ndarray] = {}
                if centroids is not None and len(centroids) > 0:
                    for i, label in enumerate(labels):
                        if i < len(centroids):
                            embeddings[str(label)] = centroids[i].flatten()

                logger.info("Diarized with embeddings: %d turns, %d speaker centroids",
                           len(turns), len(embeddings))
                return turns, embeddings
            else:
                # 구버전 pyannote — return_embeddings 미지원 fallback
                logger.info("return_embeddings not supported, falling back to separate extraction")
                annotation = getattr(result, "speaker_diarization", result)
        except TypeError:
            # return_embeddings 파라미터를 지원하지 않는 구버전
            logger.info("return_embeddings parameter not supported, using fallback")
            result = pipeline(audio_input)
            annotation = getattr(result, "speaker_diarization", result)

        turns = []
        for seg, _, speaker in annotation.itertracks(yield_label=True):
            turns.append(DiarTurn(speaker=str(speaker), start=float(seg.start), end=float(seg.end)))

        # fallback: 별도 임베딩 추출
        embeddings = extract_speaker_embeddings(waveform, sample_rate, turns)
        return turns, embeddings

    # return_embeddings=False: 기존 동작
    result = pipeline(audio_input)
    annotation = getattr(result, "speaker_diarization", result)

    turns = []
    for seg, _, speaker in annotation.itertracks(yield_label=True):
        turns.append(DiarTurn(speaker=str(speaker), start=float(seg.start), end=float(seg.end)))
    return turns


@dataclass
class DiarEpochResult:
    """한 에폭(10분 청크)의 diarization 결과."""
    epoch_idx: int
    offset_sec: float  # 전체 오디오 내에서의 시작 위치
    duration_sec: float
    turns: list[DiarTurn] = field(default_factory=list)
    # speaker_id → embedding vector (numpy array serialized as list)
    speaker_embeddings: dict[str, list[float]] = field(default_factory=dict)
    # local speaker_id → global speaker_id 매핑
    speaker_map: dict[str, str] = field(default_factory=dict)


def extract_speaker_embeddings(
    waveform,
    sample_rate: int,
    turns: list[DiarTurn],
    device: str = "cpu",
) -> dict[str, np.ndarray]:
    """각 화자의 발화 구간에서 임베딩 벡터를 추출한다.

    각 화자의 모든 발화 구간 임베딩을 평균하여 대표 벡터를 만든다.
    """
    inference = get_embedding_inference(device)
    if inference is None:
        return {}

    from pyannote.core import Segment

    # 화자별 발화 구간 수집
    speaker_segments: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for t in turns:
        if t.end - t.start >= 0.5:  # 최소 0.5초 이상의 구간만
            speaker_segments[t.speaker].append((t.start, t.end))

    embeddings: dict[str, np.ndarray] = {}

    for speaker, segments in speaker_segments.items():
        speaker_embeds = []
        for start, end in segments:
            try:
                # waveform에서 해당 구간 크롭
                start_sample = int(start * sample_rate)
                end_sample = min(int(end * sample_rate), waveform.shape[-1])
                if end_sample - start_sample < sample_rate * 0.5:
                    continue

                segment_wav = waveform[:, start_sample:end_sample]
                emb = inference({"waveform": segment_wav, "sample_rate": sample_rate})
                if emb is not None:
                    speaker_embeds.append(emb.flatten())
            except Exception as e:
                logger.debug("Embedding extraction failed for %s [%.1f-%.1f]: %s",
                            speaker, start, end, e)
                continue

        if speaker_embeds:
            # 모든 구간의 임베딩을 평균하여 대표 벡터 생성
            embeddings[speaker] = np.mean(speaker_embeds, axis=0)
            logger.debug("Speaker %s: averaged %d segment embeddings", speaker, len(speaker_embeds))

    return embeddings


def match_speakers_by_embedding(
    prev_embeddings: dict[str, np.ndarray],
    curr_embeddings: dict[str, np.ndarray],
    threshold: float = 0.65,
) -> dict[str, str]:
    """임베딩 cosine similarity로 이전/현재 에폭의 화자를 매칭한다.

    Returns:
        curr_local_speaker → prev_global_speaker 매핑
    """
    if not prev_embeddings or not curr_embeddings:
        return {}

    from scipy.spatial.distance import cdist

    prev_speakers = list(prev_embeddings.keys())
    curr_speakers = list(curr_embeddings.keys())

    prev_matrix = np.stack([prev_embeddings[s] for s in prev_speakers])
    curr_matrix = np.stack([curr_embeddings[s] for s in curr_speakers])

    # cosine distance → similarity (1 - distance)
    dist_matrix = cdist(curr_matrix, prev_matrix, metric="cosine")
    sim_matrix = 1.0 - dist_matrix

    # 탐욕적 1:1 매핑 (similarity 높은 순)
    mapping: dict[str, str] = {}
    used_prev: set[str] = set()

    # 모든 (curr, prev) 쌍을 similarity 높은 순으로 정렬
    pairs = []
    for ci, cs in enumerate(curr_speakers):
        for pi, ps in enumerate(prev_speakers):
            pairs.append((sim_matrix[ci, pi], cs, ps))
    pairs.sort(key=lambda x: -x[0])

    for sim, curr_spk, prev_spk in pairs:
        if curr_spk in mapping or prev_spk in used_prev:
            continue
        if sim >= threshold:
            mapping[curr_spk] = prev_spk
            used_prev.add(prev_spk)
            logger.info("Embedding match: %s → %s (similarity=%.3f)", curr_spk, prev_spk, sim)

    return mapping


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
                    global_start = max(global_start, prev_chunk_end - DIAR_OVERLAP_SEC / 2)
                    if global_start >= global_end:
                        continue

            all_turns.append(DiarTurn(speaker=global_speaker, start=global_start, end=global_end))

    return sorted(all_turns, key=lambda t: t.start)


class DiarizationClient:
    """화자분리 클라이언트 (pyannote.audio).

    싱글턴 파이프라인 + 동기 diarize 메서드.
    인크리멘탈 모드: 10분 에폭 단위 diarization + 임베딩 기반 화자 매칭.
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

    def diarize_epoch(
        self, pcm_data: bytes, sample_rate: int, offset_sec: float = 0.0,
        device: str = "cpu",
    ) -> tuple[list[DiarTurn], dict[str, np.ndarray]]:
        """단일 에폭(10분 청크)을 diarize하고 화자 임베딩도 추출한다.

        pyannote 3.0+: return_embeddings=True로 파이프라인 한 번 실행으로
        diarization + centroid 임베딩을 함께 추출한다.
        구버전: 별도 임베딩 추출로 fallback.

        Args:
            pcm_data: int16 PCM 바이트
            sample_rate: 샘플레이트
            offset_sec: 전체 오디오 내 이 청크의 시작 위치 (초)
            device: 디바이스

        Returns:
            (turns, speaker_embeddings) 튜플.
            turns는 에폭 내 로컬 시간 기준 (0부터 시작).
            speaker_embeddings는 {speaker_id: embedding_vector} 딕셔너리.
        """
        import torch

        audio = np.frombuffer(pcm_data, dtype=np.int16).astype(np.float32) / 32768.0
        duration_sec = len(audio) / sample_rate

        if len(audio) < sample_rate * 0.5:
            return [], {}

        logger.info("Diarize epoch: offset=%.1fs, duration=%.1fs", offset_sec, duration_sec)
        waveform = torch.from_numpy(audio).unsqueeze(0)

        # return_embeddings=True → 파이프라인에서 centroid 임베딩 직접 반환
        result = _diarize_waveform(waveform, sample_rate, return_embeddings=True)
        if isinstance(result, tuple):
            turns, embeddings = result
        else:
            turns = result
            embeddings = {}

        return turns, embeddings
