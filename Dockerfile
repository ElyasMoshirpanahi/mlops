# Base image
FROM python:3.11

# Set working directory
WORKDIR /app

# Copy requirements file
COPY requirements.txt .

# Install dependencies
RUN pip install -r requirements.txt

# Copy project code
COPY . .

# Export port
EXPOSE 8000 

# Run command
CMD ["uvicorn", "deployment.main:app", "--host", "0.0.0.0", "--port", "8000"]
