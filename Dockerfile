FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel

RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
# Unsloth/transformers deps can install an older torch; torchao (via transformers) needs
# torch>=2.5 with torch.utils._pytree.register_constant.
# unsloth_zoo upgrade can pull torch back down — reinstall CUDA torch again after it.
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir --upgrade torch --index-url https://download.pytorch.org/whl/cu124 \
    && pip install --no-cache-dir --upgrade unsloth_zoo \
    && pip install --no-cache-dir --upgrade torch --index-url https://download.pytorch.org/whl/cu124 \
    && python -c "import torch; assert hasattr(torch.utils._pytree, 'register_constant'), torch.__version__"

COPY . .

RUN mkdir -p /app/output

EXPOSE 7860

CMD ["python", "app.py"]
