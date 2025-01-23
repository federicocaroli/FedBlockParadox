# Use an official Python base image
FROM python:3.10-slim

# Set the working directory inside the container
WORKDIR /usr/src/app

# Copy the current directory into the container
COPY . .

# Install system dependencies, including procps for the ps command
RUN apt-get update && apt-get install -y --no-install-recommends \
    procps \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Create the /output directory
RUN mkdir -p /output

# Install Python dependencies from requirements.txt
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir --extra-index-url https://pypi.nvidia.com -r requirements.txt

# Define the entry point for the container
# IMPORTANT: config.json must be mounted into the container at runtime!
CMD ["sh", "-c", "python -m src.main config.json 2>/output/info.txt 1>/output/info.txt"]