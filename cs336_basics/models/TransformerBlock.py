import torch
import torch.nn as nn
from torch import Tensor
import cs336_basics.models.basic as basic

class TransformerBlock(nn.Module):

    def __init__(
        self, 
        d_model: int, 
        num_heads: int, 
        d_ff: int, 
        max_seq_len: int, 
        rope_theta: float, 
        device: str = "cpu", 
        dtype: torch.dtype = torch.float32):

        super().__init__()
        self.ln1 = basic.RMSNorm(d_model, device=device, dtype=dtype)
        self.attn = basic.MultiHeadAttentionWithRope(d_model, num_heads, max_seq_len, rope_theta, device=device, dtype=dtype)
        self.ln2 = basic.RMSNorm(d_model, device=device, dtype=dtype)
        self.ffn = basic.SwiGLU(d_model, d_ff, device=device, dtype=dtype)

        self.d_model = d_model
        self.d_ff = d_ff
        self.rope_theta = rope_theta
        self.context_length = max_seq_len
        self.num_heads = num_heads

    def forward(self, in_features):
        x_1 = self.ln1(in_features)
        x_attn = self.attn(x_1, token_positions=torch.arange(in_features.shape[1], device=in_features.device))
        x_1 = x_attn + in_features 

        x_2 = self.ln2(x_1)
        x_2 = self.ffn(x_2)
        x_2 = x_2 + x_1 
        return x_2