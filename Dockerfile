FROM nvcr.io/nvidia/pytorch:25.12-py3

WORKDIR /workspace

RUN python -m pip install --no-cache-dir --upgrade pip setuptools wheel

RUN python -m pip install --no-cache-dir \
    huggingface_hub \
    transformers \
    datasets \
    accelerate \
    peft \
    trl \
    bitsandbytes \
    wandb \
    scipy

CMD ["bash"]

