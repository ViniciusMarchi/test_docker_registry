FROM ghcr.io/mlflow/mlflow:latest

COPY docker/mlflow/requirements.txt .

# Install system dependencies for psycopg2
RUN apt-get update && apt-get install -y \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

RUN pip install -r requirements.txt