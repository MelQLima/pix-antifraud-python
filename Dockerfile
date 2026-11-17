FROM python:3.12-slim AS build
WORKDIR /app
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip wheel --wheel-dir /wheels .

FROM python:3.12-slim
WORKDIR /app
RUN useradd --create-home --uid 10001 app
COPY --from=build /wheels /wheels
RUN pip install --no-cache-dir /wheels/* && rm -rf /wheels
USER app
ENV ANTIFRAUDE_MODEL_DIR=/app/models
EXPOSE 8000
HEALTHCHECK CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"
CMD ["uvicorn", "antifraude.api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
