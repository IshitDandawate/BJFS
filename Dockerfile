FROM python:3.9-slim-buster

# Set working directory
WORKDIR /app

# Install system dependencies for Tesseract and OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements.txt and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY . .

# Set environment variables (if needed)
# ENV TESSDATA_PREFIX=/usr/share/tesseract-ocr/tessdata

# Expose the port that Uvicorn will listen on
EXPOSE 8000

# Command to run the FastAPI application using Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]