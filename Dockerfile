# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Set environment variable for pip timeout
ENV PIP_DEFAULT_TIMEOUT=300

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libomp-dev \
    wget \
    git \
    python3-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry for dependency management
RUN curl -sSL https://install.python-poetry.org | python3 -

# Make Poetry available in the PATH
ENV PATH="$PATH:/root/.local/bin"

# Copy Poetry files first (for better caching)
COPY pyproject.toml poetry.lock /app/

# Copy the rest of the application
COPY . /app/

# Copy project files
COPY . /app/

# Install dependencies using Poetry
RUN poetry config virtualenvs.create false
RUN poetry install --no-interaction --no-ansi

# Expose port 8000 for FastAPI
EXPOSE 8000

# Command to run the application
CMD ["poetry", "run", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
