# Use slim Python image for smaller size
FROM python:3.11-slim AS builder

# Set working directory
WORKDIR /app

# Copy only requirements to leverage caching
COPY requirements.txt .

# Install dependencies
RUN pip install -r requirements.txt

# Copy app code
COPY . . 

# Create unprivileged user to run process
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Build runtime image
FROM python:3.11-slim
WORKDIR /app

# Copy from builder stage
COPY --from=builder /app .

# Change to non-root user  
USER appuser

# Configure app configs/secrets
ENV DEBUG=false
ENV APP_SECRET=supersecuresecret 

# Expose port
EXPOSE 8000

# Healthcheck
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

# Run app
CMD ["uvicorn", "deployment.main:app", "--host", "0.0.0.0", "--port", "8000"]