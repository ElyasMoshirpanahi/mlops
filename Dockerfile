# Base image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy requirements
COPY requirements.txt .

# Install dependencies
RUN pip install -r requirements.txt

# Copy app code 
COPY . .

# Set PYTHONPATH 
ENV PYTHONPATH=/app

# Expose port
EXPOSE 8000 

# Run app
CMD ["uvicorn", "deployment.main:app", "--host", "0.0.0.0", "--port", "8000"]