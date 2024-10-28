# Use an official Python image as a base
FROM python:3.12-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies if needed (e.g., gcc for some Python packages)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir \
    mlflow \
    yolov8 \
    ultralytics \
    opencv-python \
    opencv-python-headless

# Copy project files into the container (adjust as necessary)
COPY . .

# Set the default command to run your application
CMD ["python", "main.py"]



