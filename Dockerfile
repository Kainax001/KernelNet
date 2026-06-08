# ─── Stage 1: dev (CPU 전용, 전처리·디버깅) ──────────────────────────────────
FROM python:3.10-slim AS dev

# OpenCV·MediaPipe 런타임 의존성
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        libegl1 \
        libgles2 \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python3 python3 /usr/local/bin/python3.10 1 \
 && update-alternatives --install /usr/bin/python  python  /usr/local/bin/python3.10 1

COPY requirements_dev.txt /tmp/requirements_dev.txt
RUN pip install --no-cache-dir -r /tmp/requirements_dev.txt

RUN mkdir -p /workspace/data /workspace/results /workspace/checkpoints /root/.kaggle

WORKDIR /workspace

# ─── Stage 2: train (CUDA 11.8, GPU 학습) ────────────────────────────────────
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04 AS train

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.10 \
        python3.10-dev \
        python3-pip \
        libgl1 \
        libglib2.0-0 \
        libegl1 \
        libgles2 \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 \
 && update-alternatives --install /usr/bin/python  python  /usr/bin/python3.10 1

RUN pip install --no-cache-dir --upgrade pip

COPY requirements_train.txt /tmp/requirements_train.txt
RUN pip install --no-cache-dir -r /tmp/requirements_train.txt

RUN mkdir -p /workspace/data /workspace/results /workspace/checkpoints /root/.kaggle

WORKDIR /workspace
