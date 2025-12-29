FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV TURBODIFFUSION_DISABLE_FLASH_ATTN=1
WORKDIR /

# System deps
RUN apt-get update && apt-get install -y \
    git \
    wget \
    ffmpeg \
    python3 \
    python3-pip \
    build-essential \
    cmake \
    ninja-build \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

RUN ln -s /usr/bin/python3 /usr/bin/python

# Python deps (NO SpargeAttn here)
COPY requirements.txt .
RUN pip install --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# --- Install SpargeAttn explicitly ---
RUN git clone https://github.com/thu-ml/SpargeAttn.git /SpargeAttn \
 && cd /SpargeAttn \
 && pip install --no-cache-dir .

RUN pip uninstall -y flash-attn || true

# TurboDiffusion
RUN git clone https://github.com/thu-ml/TurboDiffusion.git /turbodiffusion
ENV PYTHONPATH=/turbodiffusion

# Models
COPY download_models.py .
RUN python download_models.py
RUN mv checkpoints /checkpoints

# Worker
COPY inference.py rp_handler.py /

CMD ["python", "-u", "rp_handler.py"]
