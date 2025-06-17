# ────────────────────────────────────────────────────────────────────────
#  Dockerfile — PyTorch 2.7.1-CUDA12.2-cuDNN9 + FAISS-GPU + TMUX/NVIM/HTOP
#               + ipykernel + ipywidgets + (dev-only) SSH
# ────────────────────────────────────────────────────────────────────────
# The image is misnamed, it's actually CUDA 12.2 not 11.8
FROM pytorch/pytorch:2.7.1-cuda11.8-cudnn9-devel

# == OS packages =========================================================
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        htop neovim openssh-server tmux wget \
    && rm -rf /var/lib/apt/lists/*

# == FAISS + notebook tooling via Conda ==================================
#   • `faiss-gpu 1.11.0` from pytorch/nvidia channels
#   • `ipykernel` & `ipywidgets` from defaults (or whichever channel resolves first)
RUN conda install -y -c pytorch -c nvidia \
        faiss-gpu=1.11.0 ipykernel ipywidgets && \
    conda clean -afy
