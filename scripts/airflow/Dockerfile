FROM apache/airflow:2.6.3

USER root

ENV PLAYWRIGHT_BROWSERS_PATH=/opt/airflow/playwright \
    PLAYWRIGHT_VERSION=1.34.0

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libnss3 \
        libatk-bridge2.0-0 \
        libatk1.0-0 \
        libcups2 \
        libdrm2 \
        libxkbcommon0 \
        libxcomposite1 \
        libxdamage1 \
        libxfixes3 \
        libxrandr2 \
        libgbm1 \
        libpango-1.0-0 \
        libcairo2 \
        libasound2 \
        libxshmfence1 \
        libgtk-3-0 \
        libx11-6 \
        libx11-xcb1 \
        libxcb1 \
        libxext6 \
        libxrender1 \
        libfontconfig1 \
        libfreetype6 \
        libglib2.0-0 \
        libnspr4 \
        libdbus-1-3 \
    && rm -rf /var/lib/apt/lists/* \
    && mkdir -p /opt/airflow/playwright \
    && chown -R airflow: /opt/airflow/playwright

USER airflow

RUN pip install --no-cache-dir "playwright==${PLAYWRIGHT_VERSION}" requests \
    && python -m playwright install chromium
