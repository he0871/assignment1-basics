import torch
import torch.nn as nn
from torch import Tensor
import cs336_basics.models.basic as basic
from cs336_basics.models.TransformerBlock import TransformerBlock
  
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
        self.token_embeddings = basic.Embedding(vocab_size, d_model, device, dtype)

        # ===== Transformer layers =====
        self.layers = nn.ModuleList()

        for _ in range(num_layers):
            self.layers.append(
                TransformerBlock(d_model, num_heads, d_ff, context_length, rope_theta, device, dtype)
            )

        # ===== Final LayerNorm =====
        self.ln_final = basic.RMSNorm(d_model, device=device, dtype=dtype)

        # ===== LM Head =====
        self.lm_head = basic.Linear(d_model, vocab_size, device, dtype)

        #self.reset_parameters()

    def forward(self, input_ids):
        x = self.token_embeddings(input_ids)

        for layer in self.layers:
            x = layer(x)

        x = self.ln_final(x)

        logits = self.lm_head(x)
        return logits
