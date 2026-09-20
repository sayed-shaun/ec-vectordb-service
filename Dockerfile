# The PyTorch CUDA runtime image already carries torch + cuDNN, which is why
# torch sits in pyproject's optional `torch` extra rather than in
# [project.dependencies] -- installing it from PyPI here would pull a second
# ~2.5 GB copy of the CUDA libraries on top of the base. The extra is for
# workstations and CI, which have no base image to inherit it from.
#
# torch must be >= 2.6. BAAI/bge-m3 publishes only `pytorch_model.bin` with no
# safetensors, and transformers refuses to `torch.load` a pickle checkpoint on
# older torch because of CVE-2025-32434. On 2.5.1 the service dies at startup
# with "we now require users to upgrade torch to at least v2.6". Do not pin
# this back without first checking whether bge-m3 has shipped safetensors.
#
# CUDA 12.8 rather than 12.4 so the same image also covers newer server GPUs
# (Blackwell / sm_120); it runs the Ampere cards fine too.
FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/models \
    TRANSFORMERS_OFFLINE=0

WORKDIR /srv

# Dependencies are read out of pyproject and installed on their own layer,
# ahead of the source. `pip install .` would work in one line but would
# rebuild the whole ~2 GB dependency layer on every source edit, because the
# source has to be present before the project can be built.
COPY pyproject.toml .
RUN python -c "import tomllib;print('\n'.join(tomllib.load(open('pyproject.toml','rb'))['project']['dependencies']))" > /tmp/requirements.txt \
 && pip install --no-cache-dir -r /tmp/requirements.txt

COPY src ./src
COPY main.py ./
COPY scripts ./scripts

# Weights live on a named volume, not in the image: ~3.5 GB of checkpoints
# would otherwise be rebuilt into every layer push.
VOLUME ["/models"]

EXPOSE 8000

# start-period is generous on purpose: with EAGER_LOAD_MODELS the first
# boot downloads ~3.5 GB of weights into /models before the port answers.
# Subsequent boots hit the cache and come up in well under a minute.
HEALTHCHECK --interval=30s --timeout=10s --start-period=900s --retries=3 \
  CMD python -c "import urllib.request;urllib.request.urlopen('http://localhost:8000/health').read()" || exit 1

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
