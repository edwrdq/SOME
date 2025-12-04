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
        # 1. Router chooses experts
        weights = Router(self.num_experts)(h)

        # 2. Threshold
        mask = weights > self.tau
        masked_weights = weights * mask

        # 3. Apply the active experts
        outputs = []
        for i in range(self.num_experts):
            if mask[i]:
                out = Expert(self.hidden_dim, self.expert_dim)(h)
                outputs.append(masked_weights[i] * out)

        # 4. Combine
        if outputs:
            return sum(outputs)
        else:
            return jnp.zeros_like(h)
