FROM python:3.14-slim

WORKDIR /app

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

# 系统依赖（OCR / PDF 必备，不变）
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    tesseract-ocr-chi-sim \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

# 复制 uv 配置文件
COPY pyproject.toml uv.lock ./

# 安装依赖（直接用 uv，不再下载 uv）
RUN uv sync --frozen --system --no-cache

# 复制代码
COPY . .

EXPOSE 8080

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
