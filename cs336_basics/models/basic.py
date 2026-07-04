import torch
import torch.nn as nn
from jaxtyping import Bool, Float, Int
from einops import rearrange, einsum
from torch import Tensor
import math

def softmax(x: Float[Tensor, " ..."], dim: int) -> Float[Tensor, " ..."]:
    x_shifted = x - torch.max(x, dim=dim, keepdim=True).values
    #print(f"x_shifted: {x_shifted}")
    exp_x = torch.exp(x_shifted)
    return exp_x / torch.sum(exp_x, dim=dim, keepdim=True)

class Linear(nn.Module):
    def __init__(self, d_in: int, d_out: int):
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.weight = nn.Parameter(torch.empty(d_out, d_in))
        
    def forward(self, in_features: Float[Tensor, " ... d_in"]) -> Float[Tensor, " ... d_out"]:
        return in_features @ rearrange(self.weight, "d_out d_in -> d_in d_out")

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

class Embedding(nn.Module):
    def __init__(self, vocab_size: int, d_model: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.weight = nn.Parameter(torch.empty(vocab_size, d_model))
        
    def forward(self, token_ids: Int[Tensor, " ..."]) -> Float[Tensor, " ... d_model"]:
        return self.weight[token_ids] 


def silu(x: Float[Tensor, " ..."]) -> Float[Tensor, " ..."]:
    return x / (1 + torch.exp(-x))

class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff

        self.w1 = Linear(d_model, d_ff)
        self.w2 = Linear(d_ff, d_model)
        self.w3 = Linear(d_model, d_ff)

    def forward(self, in_features: Float[Tensor, " ... d_model"]) -> Float[Tensor, " ... d_model"]:
        #          [... dff]  [... dff]
        gated = silu(self.w1(in_features)) * (self.w3(in_features))
        
        return self.w2(gated)


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
        
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.q_proj = Linear(d_model, d_model)
        self.k_proj = Linear(d_model, d_model)
        self.v_proj = Linear(d_model, d_model)
        self.output_proj = Linear(d_model, d_model)
    
    def forward(self, in_features: Float[Tensor, " ... sequence_length d_model"]) -> Float[Tensor, " ... sequence_length d_model"]:
        Q = self.q_proj(in_features)
        K = self.k_proj(in_features) 
        V = self.v_proj(in_features)
        Q = rearrange(Q, " ... seq (head d_head) -> ... head seq d_head", head=self.num_heads)
        K = rearrange(K, " ... seq (head d_head) -> ... head seq d_head", head=self.num_heads)
        V = rearrange(V, " ... seq (head d_head) -> ... head seq d_head", head=self.num_heads)
        mask = torch.triu(
            torch.ones(Q.shape[-2], Q.shape[-2], dtype=torch.bool, device=self.q_proj.weight.device),
            diagonal=1
       )
        attn_scores = scaled_dot_product_attention(Q, K, V, ~mask)
        attn_scores = rearrange(attn_scores, "... head seq d_head -> ... seq (head d_head)")
        return self.output_proj(attn_scores)

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

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, in_features: Float[Tensor, " ... d_model"]) -> Float[Tensor, " ... d_model"]:
        return in_features / torch.sqrt(torch.mean(in_features**2, dim=-1, keepdim=True) + self.eps) * self.weight


class MultiHeadAttentionWithRope(nn.Module):
    def __init__(self, d_model: int, num_heads: int, max_seq_len: int, theta: float):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.max_seq_len = max_seq_len
        self.theta = theta
        
        self.q_proj = Linear(d_model, d_model)
        self.k_proj = Linear(d_model, d_model)
        self.v_proj = Linear(d_model, d_model)
        self.output_proj = Linear(d_model, d_model)
    
    def forward(self, in_features: Float[Tensor, " ... sequence_length d_model"], token_positions: Int[Tensor, " ... sequence_length"]) -> Float[Tensor, " ... sequence_length d_model"]:
        Q = self.q_proj(in_features)
        K = self.k_proj(in_features)
        V = self.v_proj(in_features)
        Q = rearrange(Q, " ... seq (head d_head) -> ... head seq d_head", head=self.num_heads)
        K = rearrange(K, " ... seq (head d_head) -> ... head seq d_head", head=self.num_heads)
        V = rearrange(V, " ... seq (head d_head) -> ... head seq d_head", head=self.num_heads)
        mask = torch.triu(
            torch.ones(Q.shape[-2], Q.shape[-2], dtype=torch.bool, device=Q.device),
            diagonal=1
        )
        rope_Q = rope(
            d_k=self.d_model // self.num_heads,
            theta=self.theta,
            max_seq_len=self.max_seq_len,
            in_query_or_key=Q,
            token_positions=token_positions,
        )
        rope_K = rope(
            d_k=self.d_model // self.num_heads,
            theta=self.theta,
            max_seq_len=self.max_seq_len,
            in_query_or_key=K,
            token_positions=token_positions,
        )
        attn_scores = scaled_dot_product_attention(rope_Q, rope_K, V, ~mask)
        attn_scores = rearrange(attn_scores, "... head seq d_head -> ... seq (head d_head)")
        return self.output_proj(attn_scores)


        

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
        torch.ones(Q.shape[-2], Q.shape[-2], dtype=torch.bool, device=Q.device),
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