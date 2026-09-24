FROM python:3.11-slim

# Install Icarus Verilog for RTL compilation/simulation
RUN apt-get update \
    && apt-get install -y --no-install-recommends iverilog \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .

ENV PYTHONUNBUFFERED=1

CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT:-8000}"]
