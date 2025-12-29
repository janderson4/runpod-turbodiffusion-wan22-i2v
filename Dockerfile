FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /

RUN apt-get update && apt-get install -y \
    git wget ffmpeg python3 python3-pip && \
    rm -rf /var/lib/apt/lists/*

RUN ln -s /usr/bin/python3 /usr/bin/python

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# TurboDiffusion source
RUN git clone https://github.com/thu-ml/TurboDiffusion.git turbodiffusion
ENV PYTHONPATH=/turbodiffusion

# Download models
COPY download_models.py .
RUN python download_models.py
RUN mv checkpoints /checkpoints

COPY inference.py rp_handler.py /

CMD ["python", "-u", "rp_handler.py"]
