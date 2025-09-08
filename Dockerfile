# Use the official Python image
FROM python:3.11-slim

# Set working directory in container
WORKDIR /app

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

RUN apt-get update && apt-get install -y curl
RUN curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
RUN apt-get install -y nodejs

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt


# Copy the rest of the application
COPY . .

# Expose the port the app runs on
EXPOSE 8051

# Command to run the application
# Note: The actual command is overridden in docker-compose.yml
CMD ["streamlit", "run", "app.py"]
