FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/.cache/huggingface \
    TRANSFORMERS_CACHE=/app/.cache/huggingface

WORKDIR /app

# libgl/glib for Pillow/opencv image handling
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.inference.txt ./requirements.inference.txt
RUN pip install --no-cache-dir \
        --extra-index-url https://download.pytorch.org/whl/cpu \
        -r requirements.inference.txt

COPY app.py setup.py README.md requirements.txt ./
COPY vision ./vision
COPY templates ./templates
COPY static ./static
RUN pip install --no-cache-dir --no-deps -e .

COPY best_meteor_model.pt ./best_meteor_model.pt

RUN mkdir -p static/uploads

EXPOSE 5000

CMD ["python", "app.py"]
