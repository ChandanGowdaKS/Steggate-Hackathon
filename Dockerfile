FROM python:3.11-slim

WORKDIR /app

# System deps for OpenCV + optional tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 libsm6 libxext6 libxrender-dev \
    ruby ruby-dev build-essential \
    && gem install zsteg --no-document \
    && pip install binwalk \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Install Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app files
COPY server.py          .
COPY security_engine.py .
COPY calibration_web.json .
COPY dashboard.html     .

# Storage directories (mounted as volumes in production)
RUN mkdir -p uploads sanitized

EXPOSE 5050

CMD ["python", "server.py"]