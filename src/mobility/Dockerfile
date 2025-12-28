# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies (needed for some Python packages)
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv (our package manager)
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.cargo/bin:$PATH"

# Copy project configuration files first (for better caching)
COPY pyproject.toml .
# If you have a lock file, copy it too (optional for now)
# COPY uv.lock .

# Install dependencies using uv
# --system installs into the system python, avoiding venv complexity inside Docker
RUN uv pip install --system --requirement pyproject.toml

# Copy the rest of the application code
COPY src/ ./src/
COPY data/ ./data/
# Note: In a real production app, we might mount models via volumes instead of copying
COPY models/ ./models/

# Set environment variables
ENV PYTHONPATH=/app/src
ENV PYTHONUNBUFFERED=1

# Define the default command to run the consumer
# (This can be overridden in docker-compose.yml)
CMD ["python", "-m", "mobility.pipeline.consumer"]