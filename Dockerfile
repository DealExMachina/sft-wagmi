FROM pytorch/pytorch:2.7.1-cuda12.6-cudnn9-devel

# Silence pip “running as root” noise in container builds (expected here).
ENV PIP_ROOT_USER_ACTION=ignore

RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
# torchao (pulled by transformers) needs torch.utils._pytree.register_constant — added in PyTorch 2.7+.
# The cu124 wheel index only ships torch up to 2.6.x, so use CUDA 12.6 base + cu126 wheels for 2.7+.
# unsloth_zoo upgrade can downgrade torch — reinstall CUDA torch again after it.
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir --upgrade torch --index-url https://download.pytorch.org/whl/cu126 \
    && pip install --no-cache-dir --upgrade unsloth_zoo \
    && pip install --no-cache-dir --upgrade torch --index-url https://download.pytorch.org/whl/cu126 \
    && python -c "import torch; assert hasattr(torch.utils._pytree, 'register_constant'), torch.__version__"

COPY . .

RUN mkdir -p /app/output

EXPOSE 7860

CMD ["python", "app.py"]
