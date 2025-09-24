FROM python:3.11-slim

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Install deps
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir fastapi uvicorn[standard] requests

# Copy source (all the split modules)
COPY *.py /app/

EXPOSE 9000

# Run the FastAPI app from main.py (object: app)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "9000"]
