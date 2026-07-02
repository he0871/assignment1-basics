import cs336_basics.models.transformer as transformer
import cs336_basics.models.worker as worker 
import cs336_basics.models.AdamW as AdamW
import cs336_basics.models.checkpoint_util as checkpoint_util
import numpy as np
import torch
import math
import matplotlib.pyplot as plt
import yaml
import time

def init_linear_weight(out_features, in_features, device=None, dtype=None):
    w = torch.empty(out_features, in_features, device=device, dtype=dtype)
    std = math.sqrt(2.0 / (in_features + out_features))
    torch.nn.init.trunc_normal_(w, mean=0.0, std=std, a=-3 * std, b=3 * std)
    return w.requires_grad_()


def init_embedding_weight(vocab_size, d_model, device=None, dtype=None):
    w = torch.empty(vocab_size, d_model, device=device, dtype=dtype)
    torch.nn.init.trunc_normal_(w, mean=0.0, std=1.0, a=-3.0, b=3.0)
    return w.requires_grad_()


def init_rmsnorm_weight(d_model, device=None, dtype=None):
    return torch.ones(d_model, device=device, dtype=dtype, requires_grad=True)


def init_linear_weight(out_features, in_features, device=None, dtype=None):
    w = torch.empty(out_features, in_features, device=device, dtype=dtype)
    std = math.sqrt(2.0 / (in_features + out_features))
    torch.nn.init.trunc_normal_(w, mean=0.0, std=std, a=-3 * std, b=3 * std)
    return w.requires_grad_()


def init_embedding_weight(vocab_size, d_model, device=None, dtype=None):
    w = torch.empty(vocab_size, d_model, device=device, dtype=dtype)
    torch.nn.init.trunc_normal_(w, mean=0.0, std=1.0, a=-3.0, b=3.0)
    return w.requires_grad_()


def init_rmsnorm_weight(d_model, device=None, dtype=None):
    return torch.ones(d_model, device=device, dtype=dtype, requires_grad=True)


def init_transformer_lm_weights(
    vocab_size,
    d_model,
    num_layers,
    num_heads,
    d_ff,
    device=None,
    dtype=torch.float32,
):
    weights = {}

    weights["token_embeddings.weight"] = init_embedding_weight(
        vocab_size, d_model, device, dtype
    )

    for layer in range(num_layers):
        prefix = f"layers.{layer}."

        weights[prefix + "ln1.weight"] = init_rmsnorm_weight(d_model, device, dtype)
        weights[prefix + "ln2.weight"] = init_rmsnorm_weight(d_model, device, dtype)

        weights[prefix + "attn.q_proj.weight"] = init_linear_weight(
            d_model, d_model, device, dtype
        )
        weights[prefix + "attn.k_proj.weight"] = init_linear_weight(
            d_model, d_model, device, dtype
        )
        weights[prefix + "attn.v_proj.weight"] = init_linear_weight(
            d_model, d_model, device, dtype
        )
        weights[prefix + "attn.output_proj.weight"] = init_linear_weight(
            d_model, d_model, device, dtype
        )

        weights[prefix + "ffn.w1.weight"] = init_linear_weight(
            d_ff, d_model, device, dtype
        )
        weights[prefix + "ffn.w2.weight"] = init_linear_weight(
            d_model, d_ff, device, dtype
        )
        weights[prefix + "ffn.w3.weight"] = init_linear_weight(
            d_ff, d_model, device, dtype
        )

    weights["ln_final.weight"] = init_rmsnorm_weight(d_model, device, dtype)

    weights["lm_head.weight"] = init_linear_weight(
        vocab_size, d_model, device, dtype
    )

    return weights

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

    weights = init_transformer_lm_weights(
    vocab_size=vocab_size,
    d_model=d_model,
    num_layers=num_layers,
    num_heads=num_heads,
    d_ff=d_ff,
    device=device,
)
    optimizer = AdamW.adamw(weights.values(), lr=learning_rate)

    dataset = np.memmap(
    "data/tinystories_train_encoded.txt",
    dtype=np.uint16,
    mode="r",
    )
chunk_tokens = 1_000_000

losses = []

start_time = time.time()
for start in range(0, len(dataset), chunk_tokens):
    chunk = dataset[start:start + chunk_tokens]

    starts = np.random.randint(
        0,
        len(chunk) - context_length - 1,
        size=batch_size,
    )

    x = np.stack([
        chunk[s:s+context_length]
        for s in starts
    ])

    y = np.stack([
        chunk[s+1:s+context_length+1]
        for s in starts
    ])

    x = torch.from_numpy(x).long().to(device)
    y = torch.from_numpy(y).long().to(device)

    logits = transformer.transformer_lm(
            vocab_size,
            context_length,
            d_model,
            num_layers,
            num_heads,
            d_ff,
            rope_theta,
            weights,
            x,
        )
    #logits is (batch_size, sequence_length, vocab_size)
    #print(logits.shape)
    loss = worker.cross_entropy(logits.view(-1, vocab_size), y.view(-1))
    print(loss)
    losses.append(loss.item())
    loss.backward()
    worker.gradient_clipping(weights.values(), 1.0)
    optimizer.step()
    optimizer.zero_grad()

plt.plot(losses)
plt.savefig("losses.png")

checkpoint_util.save_checkpoint(weights, optimizer, 0, "cs336_basics/train_lm/tingStories.pt")
    
end_time = time.time()
print(f"Time taken: {end_time - start_time} seconds")