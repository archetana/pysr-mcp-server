# Use official Python runtime as base image
FROM python:3.11-slim

# Set working directory in container
WORKDIR /app

# Install system dependencies required for building scientific packages and Julia
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    wget \
    curl \
    ca-certificates \
    libopenblas-dev \
    liblapack-dev \
    gfortran \
    libgomp1 \
    unzip \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Install Julia (1.9.x) — download and extract into /opt, symlink to /usr/local/bin
ARG JULIA_VERSION=1.9.3
RUN wget -qO /tmp/julia.tar.gz https://julialang-s3.julialang.org/bin/linux/x64/1.9/julia-${JULIA_VERSION}-linux-x86_64.tar.gz \
    && tar -xzf /tmp/julia.tar.gz -C /opt \
    && rm /tmp/julia.tar.gz \
    && ln -s /opt/julia-${JULIA_VERSION}/bin/julia /usr/local/bin/julia

# Copy requirements and install Python dependencies
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy application files
COPY pysr_mcp_server.py /app/pysr_mcp_server.py
COPY proxy_server.py /app/proxy_server.py

# Create directories for models, data, and results
RUN mkdir -p /app/models /app/data /app/results

# Initialize PySR Julia backend (this will download Julia packages for PySR)
# Do this after Julia and pysr are installed. It's okay if it runs as root in a container.
RUN python -c "import pysr; pysr.install()"

# Expose port for the MCP server
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c \"import requests; requests.get('http://localhost:8000/health')\" || exit 1

# Default command to run the server
CMD ["python", "pysr_mcp_server.py"]