# src/model/expert.py
import flax.linen as nn
import jax.numpy as jnp

class Expert(nn.Module):
    hidden_dim: int
    expert_dim: int

    @nn.compact
    def __call__(self, h):
        x = nn.Dense(self.expert_dim)(h)
        x = nn.gelu(x)
        x = nn.Dense(self.hidden_dim)(x)
        return x
