import time

import numpy as np
from transformers import AutoTokenizer, GPTNeoXForCausalLM


def test_pytorch():
    import torch

    print(f"PyTorch: {torch.__version__} | CUDA: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        x = torch.randn(2000, 2000, device="cuda")
        y = torch.randn(2000, 2000, device="cuda")

        start = time.time()
        z = torch.matmul(x, y)
        gpu_time = time.time() - start

        print(f"GPU matmul: {gpu_time:.3f}s ✅")
        return True
    return False


def test_faiss():
    import faiss

    print(f"Faiss: {faiss.__version__} | GPUs: {faiss.get_num_gpus()}")

    if faiss.get_num_gpus() > 0:
        # Create test data
        vectors = np.random.random((10000, 128)).astype("float32")
        query = np.random.random((100, 128)).astype("float32")

        # GPU index
        gpu_res = faiss.StandardGpuResources()
        index = faiss.IndexFlatL2(128)
        gpu_index = faiss.index_cpu_to_gpu(gpu_res, 0, index)
        gpu_index.add(vectors)

        start = time.time()
        distances, indices = gpu_index.search(query, 5)
        gpu_time = time.time() - start

        print(f"GPU search: {gpu_time:.3f}s ✅")
        return True
    return False


def test_transformers():
    import requests

    try:
        response = requests.get("https://huggingface.co/EleutherAI/pythia-14m")
        print("Status:", response.status_code)
    except Exception as e:
        print("Connection error:", e)

    model_str = "EleutherAI/pythia-14m"
    step = 5000

    model = GPTNeoXForCausalLM.from_pretrained(
        model_str, revision=f"step{step}", device_map="auto", force_download=True
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_str,
        revision=f"step{step}",
    )
    print("loaded")


if __name__ == "__main__":
    # print("🧪 Quick GPU Test")
    # print("=" * 30)

    # try:
    #     pytorch_ok = test_pytorch()
    # except Exception as e:
    #     print(f"PyTorch: ❌ {e}")
    #     pytorch_ok = False

    # try:
    #     faiss_ok = test_faiss()
    # except Exception as e:
    #     print(f"Faiss: ❌ {e}")
    #     faiss_ok = False

    # print("=" * 30)
    # if pytorch_ok and faiss_ok:
    #     print("🎉 All GPU tests passed!")
    # else:
    #     print("⚠️  Some tests failed")

    print("=" * 50)

    test_transformers()
