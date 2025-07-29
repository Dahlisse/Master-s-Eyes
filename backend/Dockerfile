FROM python:3.11-slim

# Set environment variables

ENV PYTHONUNBUFFERED=1   
PYTHONDONTWRITEBYTECODE=1   
PIP_NO_CACHE_DIR=1   
PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies

RUN apt-get update && apt-get install -y   
build-essential   
curl   
libpq-dev   
libffi-dev   
libssl-dev   
git   
&& rm -rf /var/lib/apt/lists/*

# Install Java (required for KoNLPy)

RUN apt-get update && apt-get install -y   
default-jdk   
&& rm -rf /var/lib/apt/lists/*

# Set JAVA_HOME

ENV JAVA_HOME=/usr/lib/jvm/default-java

# Create app directory

WORKDIR /app

# Copy requirements first (for better caching)

COPY requirements.txt .

# Install Python dependencies

RUN pip install –upgrade pip &&   
pip install -r requirements.txt

# Install KoNLPy data

RUN python -c “import nltk; nltk.download(‘punkt’)”

# Copy application code

COPY . .

# Create logs directory

RUN mkdir -p /app/logs

# Expose port

EXPOSE 8000

# Health check

HEALTHCHECK –interval=30s –timeout=30s –start-period=5s –retries=3   
CMD curl -f http://localhost:8000/health || exit 1

# Default command (can be overridden in docker-compose)

CMD [“uvicorn”, “app.main:app”, “–host”, “0.0.0.0”, “–port”, “8000”]