FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    wget \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /opt/render/project/src

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY main.py .

# Create temp directory
RUN mkdir -p /tmp/video_clips

EXPOSE $PORT

CMD uvicorn main:app --host 0.0.0.0 --port $PORT
