import cs336_basics.models.TransformerLM as TransformerLM
import cs336_basics.models.worker as worker 
import cs336_basics.models.AdamW as AdamW
import cs336_basics.models.checkpoint_util as checkpoint_util
import numpy as np
import torch
import math
import matplotlib.pyplot as plt
import yaml
import time
import torch.nn.functional as F


if __name__ == "__main__":
    with open("cs336_basics/train_lm/hyconfig.yaml", "r") as f:
        config = yaml.safe_load(f)
    batch_size = config["batch_size"]
    context_length = config["context_length"]
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    #device = config["device"]
    load_size = config["load_size"]
    vocab_size = config["vocab_size"]
    d_model = config["d_model"]
    num_layers = config["num_layers"]
    num_heads = config["num_heads"]
    d_ff = config["d_ff"]
    rope_theta = config["rope_theta"]
    learning_rate = float(config["learning_rate"])
    dtype =torch.float32

    model = TransformerLM.TransformerLM(vocab_size, context_length, d_model, num_layers, num_heads, d_ff, rope_theta, device, dtype)

    optimizer = AdamW.adamw(model.parameters(), lr=learning_rate)

    dataset = np.memmap(
    "data/tinystories_train_encoded.txt",
    dtype=np.uint16,
    mode="r",
    )
    #chunk_tokens = 1_000_000
    chunk_tokens = 200_000
    losses = []

    start_time = time.time()
    for start in range(0, len(dataset), chunk_tokens):
        chunk = dataset[start:start + chunk_tokens]

        starts = np.random.randint(
            0,
            len(chunk) - context_length - 1,
            size=batch_size,
        )

        x = torch.from_numpy(np.stack([
            chunk[s:s+context_length]
            for s in starts
        ])).long().to(device)

        y = torch.from_numpy(np.stack([
            chunk[s+1:s+context_length+1]
            for s in starts
        ])).long().to(device)

        logits = model(x)
        #logits is (batch_size, sequence_length, vocab_size)
        #print(logits.shape)
        loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))
        print(loss.item())
        losses.append(loss.item())
        loss.backward()
        worker.gradient_clipping(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()

    plt.plot(losses)
    plt.title(f"Losses lr={learning_rate}")
    plt.savefig("losses.png")

    checkpoint_util.save_checkpoint(model.state_dict(), optimizer, 0, "cs336_basics/train_lm/tinyStories.pt")
    
    end_time = time.time()
    print(f"Time taken: {end_time - start_time} seconds")