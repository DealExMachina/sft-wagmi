FROM pytorch/pytorch:2.7.1-cuda12.6-cudnn9-devel

# Silence pip “running as root” noise in container builds (expected here).
ENV PIP_ROOT_USER_ACTION=ignore

RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*

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

RUN mkdir -p /app/output

EXPOSE 7860

CMD ["python", "app.py"]
