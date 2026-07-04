FROM python:3.12-slim

WORKDIR /app

COPY pyproject.toml README.md /app/
COPY llm_guard /app/llm_guard
COPY config /app/config

RUN pip install --no-cache-dir .

EXPOSE 8000

CMD ["uvicorn", "llm_guard.app:app", "--host", "0.0.0.0", "--port", "8000"]
