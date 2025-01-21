FROM python:3.12-slim

# Install system dependencies if needed
# RUN apt-get update && apt-get install -y --no-install-recommends \
#     gcc \
#     && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy the Python dependencies and install
COPY docker/data_ingestion/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the ingestion script into the container
COPY src/data_ingestion/data_ingestion.py .

CMD ["python", "data_ingestion.py"]
