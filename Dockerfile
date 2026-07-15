# Serving image (API layer only — vLLM runs as a separate service, same
# pattern as gvc-local). Multi-stage keeps the runtime slim.
FROM python:3.11-slim AS builder
WORKDIR /build
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir --prefix=/install ".[serve]"

FROM python:3.11-slim
RUN useradd --create-home appuser
COPY --from=builder /install /usr/local
USER appuser
ENV CRL_BASE_URL=http://vllm:8000/v1
EXPOSE 8080
CMD ["uvicorn", "connections_rl.serve.app:create_app", "--factory", "--host", "0.0.0.0", "--port", "8080"]
