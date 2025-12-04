# src/model/attention.py
import flax.linen as nn
import jax.numpy as jnp

class TinySelfAttention(nn.Module):
    hidden_dim: int

    @nn.compact
    def __call__(self, x):
        B, T, D = x.shape
        
        q = nn.Dense(D)(x)
        k = nn.Dense(D)(x)
        v = nn.Dense(D)(x)

        attn_weights = jnp.einsum("btd,bSd->btS", q, k) / jnp.sqrt(D)
        attn_weights = nn.softmax(attn_weights)

        out = jnp.einsum("btS,bSd->btd", attn_weights, v)
        return out
