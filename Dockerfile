FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates python3 python3-pip python3-venv \
    ffmpeg libsndfile1 git curl dos2unix \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/
RUN python3 -m pip install --upgrade pip \
    && python3 -m pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu124 torch torchaudio \
    && python3 -m pip install --no-cache-dir -r requirements.txt

COPY . /app

# Fix CRLF in entrypoint and any shell scripts (prevents: env: 'bash\r': No such file or directory)
RUN dos2unix /app/entrypoint.sh && chmod +x /app/entrypoint.sh \
 && find /app -type f -name "*.sh" -print0 | xargs -0 -r dos2unix

RUN mkdir -p /app/data/audio

ENV HF_HUB_OFFLINE=1
ENV TRANSFORMERS_OFFLINE=1
ENV DIAR_DEVICE=auto

ENTRYPOINT ["/app/entrypoint.sh"]
CMD ["api"]
