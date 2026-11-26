FROM python:3.11-slim

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source code and models
COPY . .

# Expose web service port
EXPOSE 8080

ENV PORT=8080
ENV HOST=0.0.0.0

# Launch the Sentinel-Chain NiceGUI console
CMD ["python", "app.py"]
