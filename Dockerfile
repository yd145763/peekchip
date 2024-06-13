# Example Dockerfile snippet
FROM python:3.8-slim

# Install system dependencies including libgl1-mesa-glx
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Expose necessary ports if any
EXPOSE 8501

# Command to run the application
CMD ["streamlit", "run", "streamlit.py"]
