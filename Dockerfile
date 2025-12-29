FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the app
COPY . .

# Expose the port
EXPOSE 8501

# Run the app
ENTRYPOINT ["streamlit", "run", "src/dashboard/Home.py", "--server.port=8501", "--server.address=0.0.0.0"]