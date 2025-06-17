# ────────────────────────────────────────────────────────────────────────
#  Dockerfile — PyTorch 2.7.1-CUDA12.6-cuDNN9 + FAISS-GPU + TMUX/NVIM/HTOP
#               + ipykernel + ipywidgets + (dev-only) SSH
# ────────────────────────────────────────────────────────────────────────
FROM pytorch/pytorch:2.7.1-cuda12.6-cudnn9-devel

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
