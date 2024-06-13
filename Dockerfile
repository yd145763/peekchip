# Use the official lightweight Python image.
# https://hub.docker.com/_/python
FROM python:3.8.19-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Install dependencies
# gcc and libgl1 are required for some of the packages to compile/install properly
RUN apt-get update && apt-get install -y \
    gcc \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Set work directory
WORKDIR /usr/src/app

# Install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the source code
COPY . .

# Expose the port Streamlit runs on
EXPOSE 8501

# Command to run the app
CMD ["streamlit", "run", "streamlit.py"]
