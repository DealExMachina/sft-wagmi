FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel

RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
# Unsloth/transformers deps can install an older torch; torchao (via transformers) needs
# torch>=2.5 with torch.utils._pytree.register_constant — reinstall CUDA wheels last.
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir --upgrade torch --index-url https://download.pytorch.org/whl/cu124

COPY . .

RUN mkdir -p /app/output

EXPOSE 7860

CMD ["python", "app.py"]
