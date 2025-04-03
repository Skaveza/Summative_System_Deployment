# Use Python 3.9 as base image
FROM python:3.9

# Set the working directory inside the container
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the container
COPY requirements.txt .

# Install dependencies
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application into the container
COPY . .

# Set the correct working directory for running the app
WORKDIR /app/src

# Command to run the application
CMD ["python", "src/app.py"]

