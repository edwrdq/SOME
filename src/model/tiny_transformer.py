# src/model/tiny_transformer.py
import flax.linen as nn
import jax.numpy as jnp
from .attention import TinySelfAttention
from .some_block import SOMEBlock

class TinyTransformerBlock(nn.Module):
    hidden_dim: int
    expert_dim: int
    num_experts: int

    @nn.compact
    def __call__(self, x):
        # Self-attention
        attn = TinySelfAttention(self.hidden_dim)(x)
        x = x + attn
        
        class TokenSOME(nn.Module):
            hidden_dim: int
            expert_dim: int
            num_experts: int

            @nn.compact
            def __call__(self, carry, h):
                y = SOMEBlock(
                    hidden_dim=self.hidden_dim,
                    expert_dim=self.expert_dim,
                    num_experts=self.num_experts,
                )(h)
                return carry, y

        scanned_some = nn.scan(
            target=TokenSOME,
            in_axes=1,
            out_axes=1,
            variable_broadcast={"params": True},
            split_rngs={"params": False},
        )(
            hidden_dim=self.hidden_dim,
            expert_dim=self.expert_dim,
            num_experts=self.num_experts,
        )

        dummy_carry = jnp.array(0, dtype=jnp.int32)
        _, some_out = scanned_some(dummy_carry, x)

        return x + some_out
