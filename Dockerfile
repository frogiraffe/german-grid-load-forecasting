FROM python:3.12-slim

WORKDIR /app

# libgomp1 is the OpenMP runtime XGBoost/LightGBM link against.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY . .
RUN pip install --no-cache-dir -e .

CMD ["python", "scripts/run_evaluate.py"]
