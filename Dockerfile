# Multi-stage build: compile the frontend, then serve it from the FastAPI
# backend as static files (SPEC.md section 14.2) — one process, one port.

FROM node:20-alpine AS frontend-build
WORKDIR /app/frontend
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

FROM python:3.12-slim AS backend
WORKDIR /app/backend

RUN pip install --no-cache-dir uv

# Install dependencies before copying the app, so code changes don't bust
# this layer's cache.
COPY backend/pyproject.toml backend/uv.lock ./
RUN uv sync --frozen --no-install-project --no-dev

COPY backend/ ./
COPY --from=frontend-build /app/frontend/dist ./static
RUN uv sync --frozen --no-dev

ENV DATA_DIR=/data
VOLUME ["/data"]

EXPOSE 8000
CMD ["uv", "run", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
