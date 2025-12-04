# src/model/router.py
import flax.linen as nn
import jax.numpy as jnp

class Router(nn.Module):
    num_experts: int

    @nn.compact
    def __call__(self, h):
        logits = nn.Dense(self.num_experts)(h)
        return nn.softmax(logits)
