# src/model/embed.py
import flax.linen as nn
import jax.numpy as jnp

class Embed(nn.Module):
    vocab_size: int
    hidden_dim: int

    @nn.compact
    def __call__(self, x):
        # x shape: (batch, seq)
        embedding = nn.Embed(self.vocab_size, self.hidden_dim)(x)
        return embedding  # (batch, seq, hidden_dim)
