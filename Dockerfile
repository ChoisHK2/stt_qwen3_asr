FROM python:3.11-slim AS base
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg git curl dos2unix && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/
RUN pip install --no-cache-dir -U pip && pip install --no-cache-dir -r requirements.txt

COPY . /app

# Fix CRLF in entrypoint and any shell scripts (prevents: env: ‘bash\r’: No such file or directory)
RUN dos2unix /app/entrypoint.sh && chmod +x /app/entrypoint.sh \
 && find /app -type f -name "*.sh" -print0 | xargs -0 -r dos2unix

ENTRYPOINT ["/app/entrypoint.sh"]
CMD ["api"]
