FROM python:3.12-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Render/free-tier: core deps only (torch/xgboost/lightgbm are heavy and
# optional — app runs in baseline mode without them). For full local ML:
#   pip install -r backend/requirements.txt
COPY backend/requirements-core.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

COPY . .

EXPOSE 8000

CMD ["python", "backend/run.py"]