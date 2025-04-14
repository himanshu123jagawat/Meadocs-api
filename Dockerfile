FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    libtesseract-dev \
    ffmpeg \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Install libvosk for vosk
RUN wget https://github.com/alphacep/vosk-api/releases/download/v0.3.45/libvosk_0.3.45-1_amd64.deb \
    && dpkg -i libvosk_0.3.45-1_amd64.deb \
    && rm libvosk_0.3.45-1_amd64.deb

# Set working directory
WORKDIR /app

# Copy requirements.txt
COPY requirements.txt .

# Install Python dependencies
RUN pip install --upgrade pip \
    && pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cpu \
    && pip install -r requirements.txt

# Copy application code
COPY . .

# Download Vosk model
RUN wget -q https://alphacephei.com/vosk/models/vosk-model-en-us-0.42-gigaspeech.zip \
    && unzip vosk-model-en-us-0.42-gigaspeech.zip \
    && mv vosk-model-en-us-0.42-gigaspeech /app/vosk-model \
    && rm vosk-model-en-us-0.42-gigaspeech.zip

# Expose port
EXPOSE 8000

# Run the application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
