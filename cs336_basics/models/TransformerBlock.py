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
        self.ln1 = basic.RMSNorm(d_model)
        self.attn = basic.MultiHeadAttentionWithRope(d_model, num_heads, max_seq_len, rope_theta)
        self.ln2 = basic.RMSNorm(d_model)
        self.ffn = basic.SwiGLU(d_model, d_ff)

        self.d_model = d_model
        self.d_ff = d_ff
        self.rope_theta = rope_theta
        self.context_length = max_seq_len
        self.num_heads = num_heads

    def forward(self, in_features):
        x = self.ln1(in_features)
        x_attn = self.attn(x, token_positions=torch.arange(in_features.shape[1], device=in_features.device))
        x = x_attn + in_features 

        x = self.ln2(x)
        x = self.ffn(x)
        x = x + in_features 
        return x