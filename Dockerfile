# Dockerfile
# HuggingFace Spaces runs this container.
# Must expose port 7860 — that's the HF default.

FROM python:3.11-slim

# Set working directory inside the container
WORKDIR /app

# Copy requirements first — Docker caches this layer
# so rebuilds are fast if only your code changed
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all project files
COPY . .

# HuggingFace Spaces expects port 7860
EXPOSE 7860

# Start the FastAPI server
# --host 0.0.0.0 makes it accessible outside the container
# --port 7860 matches what HF expects
CMD ["uvicorn", "dataoncallenv.api.app:app", "--host", "0.0.0.0", "--port", "7860"]