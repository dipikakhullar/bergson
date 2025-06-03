from argparse import ArgumentParser

import faiss
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from bergson import apply_second_moments, project_grads
from bergson.data import MemmapDataset


def main():
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, default="EleutherAI/pythia-160m")
    parser.add_argument(
        "--dataset",
        type=str,
        default="/mnt/ssd-1/pile_preshuffled/standard/document.bin",
    )
    parser.add_argument("--index", type=str, default="index.faiss")
    parser.add_argument("--moments", type=str, default="second_moments.pth")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, device_map={"": "cuda:0"})
    data = MemmapDataset(args.dataset, 2049)

    moments = torch.load(
        "second_moments.pth",
        map_location="cuda:0",
        weights_only=True,
    )

    # Load the index
    index = faiss.read_index(args.index)

    # Query loop
    while True:
        query = input("Enter your query: ")
        if query.lower() == "exit":
            break

        # Tokenize the query
        inputs = tokenizer(query, return_tensors="pt").to("cuda:0")
        x = inputs["input_ids"].unsqueeze(0)
        model(x, labels=x).loss.backward()

        apply_second_moments(model, moments)
        grad = project_grads(model)
        model.zero_grad()

        # Perform the search
        dists, indices = index.search(grad[None].cpu().numpy(), k=1)

        # Print the results
        print(f"Top 5 results for '{query}':")
        for i, (d, idx) in enumerate(zip(dists[0], indices[0])):
            tokens = data[idx]["input_ids"]
            string = tokenizer.decode(tokens, skip_special_tokens=True)
            print(f"{i + 1}: {string} (distance: {d})")


if __name__ == "__main__":
    main()
