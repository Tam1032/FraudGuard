FROM python:3.13-slim

# LightGBM needs the OpenMP runtime.
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY fraudguard/ fraudguard/
COPY sql/ sql/
COPY scripts/ scripts/
COPY tests/ tests/

ENV FRAUDGUARD_HOME=/app
EXPOSE 8000
CMD ["uvicorn", "fraudguard.api:app", "--host", "0.0.0.0", "--port", "8000"]
