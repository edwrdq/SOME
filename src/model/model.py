# src/model/model.py
import flax.linen as nn
import jax.numpy as jnp
from .embed import Embed
from .tiny_transformer import TinyTransformerBlock

class CopyModel(nn.Module):
    vocab_size: int
    hidden_dim: int
    expert_dim: int
    num_experts: int

    @nn.compact
    def __call__(self, x):
        emb = Embed(self.vocab_size, self.hidden_dim)(x)
        out = TinyTransformerBlock(
            self.hidden_dim,
            self.expert_dim,
            self.num_experts
        )(emb)

        logits = nn.Dense(self.vocab_size)(out)
        return logits
