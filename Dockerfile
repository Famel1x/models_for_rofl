FROM python:3.10-slim

RUN apt-get update && apt-get install -y \
    build-essential \
    libatlas-base-dev \
    libprotobuf-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . /app
WORKDIR /app

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "5000"]