import torch
import torch.nn as nn
from jaxtyping import Bool, Float, Int
from einops import rearrange, einsum
from torch import Tensor
import cs336_basics.models.basic as basic

def transformer_block(
    d_model: int,
    num_heads: int,
    d_ff: int,
    max_seq_len: int,
    theta: float,
    weights: dict[str, Tensor],
    in_features: Float[Tensor, " batch sequence_length d_model"],
) -> Float[Tensor, " batch sequence_length d_model"]:

    norm_in_features = basic.rmsnorm(
        d_model=d_model,
        weights=weights["ln1.weight"],
        in_features=in_features,
    )

    
    x_attn = basic.multihead_self_attention_with_rope( # shape: (batch, sequence_length, d_model)
        d_model=d_model,
        num_heads=num_heads,
        theta=theta,
        max_seq_len=max_seq_len,
        q_proj_weight=weights["attn.q_proj.weight"],
        k_proj_weight=weights["attn.k_proj.weight"],
        v_proj_weight=weights["attn.v_proj.weight"],
        o_proj_weight=weights["attn.output_proj.weight"],
        in_features=norm_in_features,
        token_positions=torch.arange(in_features.shape[1]),
    )

    in_features = x_attn + in_features # shape: (batch, sequence_length, d_model)

    norm_attn = basic.rmsnorm(
        d_model=d_model,
        weights=weights["ln2.weight"],
        in_features=in_features,
    )

    swiglu_attn = basic.swiglu(
        d_model=d_model,
        d_ff=d_ff,
        w1_weight=weights["ffn.w1.weight"],
        w2_weight=weights["ffn.w2.weight"],
        w3_weight=weights["ffn.w3.weight"],
        in_features=norm_attn,
    )

    return swiglu_attn + in_features
  
   
def transformer_lm(
    vocab_size: int,
    context_length: int,
    d_model: int,
    num_layers: int,
    num_heads: int,
    d_ff: int,
    rope_theta: float,
    weights: dict[str, Tensor],
    in_indices: Int[Tensor, " batch_size sequence_length"],
) -> Float[Tensor, " batch_size sequence_length vocab_size"]:
    
    #print(weights.keys())
    in_features = basic.embedding(
        vocab_size=vocab_size,
        d_model=d_model,
        weights=weights["token_embeddings.weight"],
        token_ids=in_indices,
    )

    for layer in range(num_layers):

        layer_weights = {k.replace(f"layers.{layer}.", ""): v for k, v in weights.items() if f"layers.{layer}." in k}
        #print(layer_weights.keys())

        in_features = transformer_block(
            d_model=d_model,
            num_heads=num_heads,
            d_ff=d_ff,
            max_seq_len=context_length,
            theta=rope_theta,
            weights=layer_weights,
            in_features=in_features,
        )

    final_features = basic.rmsnorm(
        d_model=d_model,
        weights=weights["ln_final.weight"],
        in_features=in_features,
    )

    out_features = final_features @ weights["lm_head.weight"].T


    return out_features