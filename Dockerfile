FROM pytorch/pytorch:2.7.1-cuda12.6-cudnn9-devel

# Silence pip “running as root” noise in container builds (expected here).
ENV PIP_ROOT_USER_ACTION=ignore
ENV DEBIAN_FRONTEND=noninteractive
# Triton (torchao import chain) must not use /.triton when HOME is / (common on HF Spaces).
ENV CACHE_BASE_DIR=/data
ENV TRITON_CACHE_DIR=/tmp/triton_cache
ENV XDG_CACHE_HOME=/tmp/.cache
ENV HOME=/tmp
ENV UNSLOTH_LLAMA_CPP_PATH=/opt/llama.cpp

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    build-essential \
    cmake \
    curl \
    libcurl4-openssl-dev \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Build llama.cpp at image build-time so Unsloth GGUF export never needs runtime apt/install prompts.
RUN git clone --depth 1 https://github.com/ggml-org/llama.cpp "${UNSLOTH_LLAMA_CPP_PATH}" \
    && cmake -S "${UNSLOTH_LLAMA_CPP_PATH}" -B "${UNSLOTH_LLAMA_CPP_PATH}/build" \
       -DBUILD_SHARED_LIBS=OFF -DGGML_CUDA=ON -DLLAMA_CURL=ON \
    && cmake --build "${UNSLOTH_LLAMA_CPP_PATH}/build" --config Release -j"$(nproc)" --target llama-quantize \
    && cp "${UNSLOTH_LLAMA_CPP_PATH}/build/bin/llama-quantize" "${UNSLOTH_LLAMA_CPP_PATH}/llama-quantize" \
    && chmod +x "${UNSLOTH_LLAMA_CPP_PATH}/llama-quantize" \
    && test -f "${UNSLOTH_LLAMA_CPP_PATH}/convert_hf_to_gguf.py"

WORKDIR /app

COPY requirements.txt .
# torchao needs torch.utils._pytree.register_constant (PyTorch 2.7+). Do NOT use bare `pip install -U torch`:
# that can jump to torch 2.11+ and breaks unsloth_zoo (torch<2.11) and torchvision/torchaudio pins.
# Keep torch/vision/audio on the same train as the base image (2.7.1 + cu126).
ARG TORCH_INDEX=https://download.pytorch.org/whl/cu126
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir \
        torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 \
        --index-url ${TORCH_INDEX} \
    && pip install --no-cache-dir --upgrade unsloth_zoo \
    && pip install --no-cache-dir \
        torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 \
        --index-url ${TORCH_INDEX} \
    && python -c "import torch; assert hasattr(torch.utils._pytree, 'register_constant'), torch.__version__"

COPY . .

COPY docker-entrypoint.sh /docker-entrypoint.sh
RUN chmod +x /docker-entrypoint.sh \
    && mkdir -p /tmp/triton_cache /tmp/.cache /app/output /data \
    && chmod 1777 /tmp \
    && chmod -R 777 /tmp/triton_cache /tmp/.cache /app/output /data

EXPOSE 7860

ENTRYPOINT ["/docker-entrypoint.sh"]
CMD ["python", "app.py"]
