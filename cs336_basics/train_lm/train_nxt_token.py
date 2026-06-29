import cs336_basics.models.transformer as transformer
import cs336_basics.models.worker as worker
import numpy as np
import torch
import math


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
    batch_size = 4
    context_length = 10
    device = "cpu"
    load_size = 1000
    vocab_size = 10000 # 10k vocab size for tiny stories, 32k vocab size for openwebtext
    d_model = 64
    num_layers = 3
    num_heads = 4
    d_ff = 128
    rope_theta = 10000.0
    weights = init_transformer_lm_weights(
    vocab_size=vocab_size,
    d_model=d_model,
    num_layers=num_layers,
    num_heads=num_heads,
    d_ff=d_ff,
    device=device,
)

    dataset = np.memmap(
    "data/tinystories_train_encoded.txt",
    dtype=np.uint16,
    mode="r",
    )
chunk_tokens = 1_000_000
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
    print(logits.shape)
    """
    with open("data/tinystories_train_encoded.txt", "rb") as f:
        text = f.read(load_size)
        dataset =text.decode("uint16")
        print(dataset)
        batches = worker.get_batch(text, batch_size, context_length, device)
        transformer.transformer_lm(vocab_size, context_length, d_model, num_layers, num_heads, d_ff, rope_theta, weights, batches)
    """