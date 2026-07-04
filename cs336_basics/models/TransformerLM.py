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
        self.ln1 = basic.rmsnorm(d_model)
        self.attn = basic.multihead_self_attention_with_rope(d_model, num_heads, max_seq_len, rope_theta)
        self.ln2 = basic.rmsnorm(d_model)
        self.ffn = basic.swiglu(d_model, d_ff)

        self.d_model = d_model
        self.d_ff = d_ff
        self.rope_theta = rope_theta
        self.context_length = max_seq_len
        self.num_heads = num_heads

    def forward(self, in_features):
        x = self.ln1(in_features)
        x_attn = self.attn(x)
        x = x_attn + in_features 

        x = self.ln2(x)
        x = self.ffn(x)
        x = x + in_features 
        return x
        
  
class TransformerLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.context_length = context_length
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.rope_theta = rope_theta

        # ===== Embedding =====
        self.token_embeddings = nn.Parameter(
            torch.empty(vocab_size, d_model, device=device, dtype=dtype)
        )

        # ===== Transformer layers =====
        self.layers = nn.ModuleList()

        for _ in range(num_layers):
            self.layers.append(
                TransformerBlock(d_model, d_ff, rope_theta, context_length, num_heads, context_length)
            )

        # ===== Final LayerNorm =====
        self.ln_final = basic.rmsnorm(d_model)

        # ===== LM Head =====
        self.lm_head = nn.Parameter(
            torch.empty(vocab_size, d_model)
        )

        self.reset_parameters()

    def forward(self, input_ids):
        x = basic.embedding(
            vocab_size=self.vocab_size,
            d_model=self.d_model,
            weights=self.token_embeddings,
            token_ids=input_ids,
        )

        for layer in self.layers:
            x = TransformerBlock(
                d_model=self.d_model,
                num_heads=self.num_heads,
                d_ff=self.d_ff,
                max_seq_len=self.context_length,
                theta=self.rope_theta,
                weights=layer,
                in_features=x,
                weights={
                "ln1.weight": layer.ln1,
                "attn.q_proj.weight": layer.q_proj,
                "attn.k_proj.weight": layer.k_proj,
                "attn.v_proj.weight": layer.v_proj,
                "attn.output_proj.weight": layer.o_proj,
                "ln2.weight": layer.ln2,
                "ffn.w1.weight": layer.w1,
                "ffn.w2.weight": layer.w2,
                "ffn.w3.weight": layer.w3,
            },
            in_features=x,
        )

        x = basic.rmsnorm(
            d_model=self.d_model,
            weights=self.ln_final,
            in_features=x,
        )

        logits = x @ self.lm_head.T
        return logits
