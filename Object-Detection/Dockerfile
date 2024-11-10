# Use a base Python image with a specific version and slim variant for a smaller image size
FROM python:3.9

# Install required libraries for OpenCV
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory in the container
WORKDIR /app

# Copy only the requirements.txt first to leverage Docker cache
COPY requirements.txt /app/

# Install only the dependencies (without copying the entire context)
RUN pip install --no-cache-dir -r requirements.txt

# Copy the deep_sort folder and Final.py into the container
COPY deep_sort /app/deep_sort
COPY Final.py /app/Final.py

# Set the default command to run your main Python script
CMD ["python", "Final.py"]
