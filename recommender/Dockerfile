FROM python:3.11-slim

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1

RUN pip install --no-cache-dir fastapi uvicorn requests

COPY app.py /app/app.py

EXPOSE 9000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "9000"]