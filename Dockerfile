FROM python:3.11-slim
WORKDIR /app

RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY server.py .
COPY treinar.py .

EXPOSE 5200
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "5200"]