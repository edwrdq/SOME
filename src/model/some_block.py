# src/model/some_block.py
import flax.linen as nn
import jax.numpy as jnp
from .router import Router
from .expert import Expert

class SOMEBlock(nn.Module):
    hidden_dim: int
    expert_dim: int
    num_experts: int
    tau: float = 0.05

    @nn.compact
    def __call__(self, h):
        weights = Router(self.num_experts)(h)
        mask = weights > self.tau
        masked = weights * mask
        denom = jnp.sum(masked, axis=-1, keepdims=True)
        safe_denom = jnp.where(denom > 0.0, denom, 1.0)
        weights_renorm = jnp.where(denom > 0.0, masked / safe_denom, masked)

        Experts = nn.vmap(
            Expert,
            variable_axes={"params": 0},
            split_rngs={"params": True},
            in_axes=None,
            out_axes=1,
            axis_size=self.num_experts,
        )
        expert_outs = Experts(self.hidden_dim, self.expert_dim)(h)
        y = jnp.einsum("be,bed->bd", weights_renorm, expert_outs)
        return y
