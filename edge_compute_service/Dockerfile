FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .

RUN apt-get update && apt-get install -y build-essential \
    && pip install --no-cache-dir -r requirements.txt

COPY llm_worker.py .
COPY ingest_data.py .

CMD ["python", "llm_worker.py"]