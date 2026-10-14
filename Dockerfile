FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \
    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
    HOST=0.0.0.0 PORT=10000 \
    TOUCHLINE_DATA_DIR=/app/deploy/assets/data \
    TOUCHLINE_MODELS_DIR=/app/deploy/assets/models \
    TOUCHLINE_HISTORY_PATH=/app/deploy/assets/history.json.gz \
    TOUCHLINE_AUTO_UPDATE=1
WORKDIR /app
COPY requirements.lock.txt requirements-hosting.txt ./
RUN pip install --no-cache-dir -r requirements-hosting.txt
RUN useradd --create-home --uid 10001 touchline
COPY --chown=touchline:touchline app.py serve.py ./
COPY --chown=touchline:touchline src ./src
COPY --chown=touchline:touchline web ./web
COPY --chown=touchline:touchline docs/transfer_fee_reference.json ./docs/transfer_fee_reference.json
COPY --chown=touchline:touchline deploy/assets ./deploy/assets
COPY --chown=touchline:touchline scripts/check_release.py ./scripts/check_release.py
USER touchline
RUN python scripts/check_release.py
EXPOSE 10000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s \
  CMD python -c "import os,urllib.request; urllib.request.urlopen('http://127.0.0.1:'+os.environ['PORT']+'/healthz',timeout=4)"
CMD ["python", "serve.py"]
