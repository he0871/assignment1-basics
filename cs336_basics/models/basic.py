import torch
import torch.nn as nn
from jaxtyping import Bool, Float, Int
from einops import rearrange, einsum
from torch import Tensor

def softmax(x: Float[Tensor, " ..."], dim: int) -> Float[Tensor, " ..."]:
    x_shifted = x - torch.max(x, dim=dim, keepdim=True).values
    #print(f"x_shifted: {x_shifted}")
    exp_x = torch.exp(x_shifted)
    return exp_x / torch.sum(exp_x, dim=dim, keepdim=True)

def linear(
    d_in: int,
    d_out: int,
    weights: Float[Tensor, " d_out d_in"],
    in_features: Float[Tensor, " ... d_in"],
) -> Float[Tensor, " ... d_out"]:

    return in_features @ rearrange(weights, "d_out d_in -> d_in d_out")


def embedding(
    vocab_size: int,
    d_model: int,
    weights: Float[Tensor, " vocab_size d_model"],
    token_ids: Int[Tensor, " ..."],
) -> Float[Tensor, " ... d_model"]:
    return weights[token_ids]

def silu(x: Float[Tensor, " ..."]) -> Float[Tensor, " ..."]:
    return x / (1 + torch.exp(-x))

def swiglu(
    d_model: int,
    d_ff: int,
    w1_weight: Float[Tensor, " d_ff d_model"],
    w2_weight: Float[Tensor, " d_model d_ff"],
    w3_weight: Float[Tensor, " d_ff d_model"],
    in_features: Float[Tensor, " ... d_model"],
) -> Float[Tensor, " ... d_model"]:
    # SwiGLU(x)=(SiLU(xWg​))⊙(xWv​)
    # Output=((SiLU(xWg​))⊙(xWv​))Wo​
    w_g = w1_weight
    w_v = w3_weight
    w_o = w2_weight
    
    
    gated = silu(in_features @ rearrange(w_g, "d_ff d_model -> d_model d_ff")) * (in_features @ rearrange(w_v, "d_ff d_model -> d_model d_ff"))
    return gated @ rearrange(w_o, "d_model d_ff -> d_ff d_model")


def scaled_dot_product_attention(
    Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys d_k"],
    V: Float[Tensor, " ... keys d_v"],
    mask: Bool[Tensor, " ... queries keys"] | None = None,
) -> Float[Tensor, " ... queries d_v"]:

    d_k = K.shape[-1]
    K_transposed = rearrange(K, " ... keys d_k -> ... d_k keys")
    scores = Q @ K_transposed / d_k**0.5 # scores shape: (..., queries, keys)
    if mask is not None:
        scores = scores.masked_fill(~mask, -float('inf'))
    scores = softmax(scores, dim=-1)
    return scores @ V


def multihead_self_attention(
    d_model: int,
    num_heads: int,
    q_proj_weight: Float[Tensor, " d_model d_model"],
    k_proj_weight: Float[Tensor, " d_model d_model"],
    v_proj_weight: Float[Tensor, " d_model d_model"],
    o_proj_weight: Float[Tensor, " d_model d_model"],
    in_features: Float[Tensor, " ... sequence_length d_model"],
) -> Float[Tensor, " ... sequence_length d_model"]:
    Q = in_features @ q_proj_weight.T # shape: (batch_size, sequence_length, d_model)
    K = in_features @ k_proj_weight.T # shape: (batch_size, sequence_length, d_model)
    V = in_features @ v_proj_weight.T # shape: (batch_size, sequence_length, d_model)

    Q = rearrange(Q, " ... seq (head d_head) -> ... head seq d_head", head=num_heads)
    K = rearrange(K, " ... seq (head d_head) -> ... head seq d_head", head=num_heads)
    V = rearrange(V, " ... seq (head d_head) -> ... head seq d_head", head=num_heads)

    mask = torch.triu(
        torch.ones(Q.shape[-2], Q.shape[-2], dtype=torch.bool),
        diagonal=1
    )
    #mask = rearrange(mask, "seq seq -> seq 1 seq 1")
    #print(mask)
    #print(f"Q shape: {Q.shape}")
    attn_scores = scaled_dot_product_attention(Q, K, V, ~mask)
    #attn_scores = softmax(attn_scores, dim=-1)
    #print(attn_scores)
    attn_scores = rearrange(attn_scores, "... head seq d_head -> ... seq (head d_head)")
    return  attn_scores @ o_proj_weight.T


def rope(
    d_k: int,
    theta: float,
    max_seq_len: int,
    in_query_or_key: Float[Tensor, " ... sequence_length d_k"],
    token_positions: Int[Tensor, " ... sequence_length"],
) -> Float[Tensor, " ... sequence_length d_k"]:
    """
    Run RoPE for a given input tensor.

    Args:
        d_k (int): Embedding dimension size for the query or key tensor.
        theta (float): RoPE parameter.
        max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
        in_query_or_key (Float[Tensor, "... sequence_length d_k"]): Input tensor to run RoPE on.
        token_positions (Int[Tensor, "... sequence_length"]): Tensor of shape (batch_size, sequence_length) with the token positions
    Returns:
        Float[Tensor, " ... sequence_length d_k"]: Tensor with RoPEd input.
    """
    assert d_k % 2 == 0

    x = in_query_or_key
    dtype = x.dtype
    device = x.device

    pair_idx = torch.arange(d_k // 2, device=device, dtype=dtype)

    freqs = theta ** (-2 * pair_idx / d_k)  # (pair,)

    angles = einsum(
        token_positions.to(dtype), # shape: (batch_size, sequence_length)
        freqs, # shape: (pair,)
        "... seq, pair -> ... seq pair",
    )

    cos = torch.cos(angles)
    sin = torch.sin(angles)

    x = rearrange(
        x,
        "... seq (pair two) -> ... seq pair two",
        two=2,
    )

    x1, x2 = x[..., 0], x[..., 1]

    out = torch.stack(
        [
            x1 * cos - x2 * sin,
            x1 * sin + x2 * cos,
        ],
        dim=-1,
    )

    out = rearrange(
        out,
        "... seq pair two -> ... seq (pair two)",
    )

    return out

def rmsnorm(
    d_model: int,
    weights: Float[Tensor, " d_model"],
    in_features: Float[Tensor, " ... d_model"],
    eps: float = 1e-5,
) -> Float[Tensor, " ... d_model"]:
    """Given the weights of a RMSNorm affine transform,
    return the output of running RMSNorm on the input features.

    Args:
        d_model (int): The dimensionality of the RMSNorm input.
        eps: (float): A value added to the denominator for numerical stability.
        weights (Float[Tensor, "d_model"]): RMSNorm weights.
        in_features (Float[Tensor, "... d_model"]): Input features to run RMSNorm on. Can have arbitrary leading
            dimensions.

    Returns:
        Float[Tensor,"... d_model"]: Tensor of with the same shape as `in_features` with the output of running
        RMSNorm of the `in_features`.
    """
    return in_features / torch.sqrt(torch.mean(in_features**2, dim=-1, keepdim=True) + eps) * weights


def multihead_self_attention_with_rope(
    d_model: int,
    num_heads: int,
    max_seq_len: int,
    theta: float,
    q_proj_weight: Float[Tensor, " d_model d_model"],
    k_proj_weight: Float[Tensor, " d_model d_model"],
    v_proj_weight: Float[Tensor, " d_model d_model"],
    o_proj_weight: Float[Tensor, " d_model d_model"],
    in_features: Float[Tensor, " ... sequence_length d_model"],
    token_positions: Int[Tensor, " ... sequence_length"] | None = None,
) -> Float[Tensor, " ... sequence_length d_model"]:
    Q = in_features @ q_proj_weight.T # shape: (batch_size, sequence_length, d_model)
    K = in_features @ k_proj_weight.T # shape: (batch_size, sequence_length, d_model)
    V = in_features @ v_proj_weight.T # shape: (batch_size, sequence_length, d_model)

    Q = rearrange(Q, " ... seq (head d_head) -> ... head seq d_head", head=num_heads)
    K = rearrange(K, " ... seq (head d_head) -> ... head seq d_head", head=num_heads)
    V = rearrange(V, " ... seq (head d_head) -> ... head seq d_head", head=num_heads)


    mask = torch.triu(
        torch.ones(Q.shape[-2], Q.shape[-2], dtype=torch.bool),
        diagonal=1
    )

    rope_Q = rope(
        d_k=d_model // num_heads,
        theta=theta,
        max_seq_len=max_seq_len,
        in_query_or_key=Q,
        token_positions=token_positions,
    )
    rope_K = rope(
        d_k=d_model // num_heads,
        theta=theta,
        max_seq_len=max_seq_len,
        in_query_or_key=K,
        token_positions=token_positions,
    )


    attn_scores = scaled_dot_product_attention(rope_Q, rope_K, V, ~mask)
    #attn_scores = softmax(attn_scores, dim=-1)
    #print(attn_scores)
    attn_scores = rearrange(attn_scores, "... head seq d_head -> ... seq (head d_head)")
    return  attn_scores @ o_proj_weight.T
    