# Clean-environment reproduction: docker build -t reorg . && docker run --rm reorg
ARG BASE=python:3.11-slim
FROM ${BASE}
RUN pip install --no-cache-dir uv==0.8.17
WORKDIR /app
COPY pyproject.toml uv.lock README.md LICENSE.md ./
COPY src ./src
RUN uv sync --frozen --extra dev
COPY . .
ENV PATH="/app/.venv/bin:${PATH}"
CMD ["bash", "-c", "pytest -q && scripts/run_all.sh"]
