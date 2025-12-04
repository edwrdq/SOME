import jax
import jax.numpy as jnp
from model.some_block import SOMEBlock

def main():
    key = jax.random.PRNGKey(0)

    # Fake hidden vector
    h = jnp.ones((64,))  # hidden_dim = 64

    block = SOMEBlock(
        hidden_dim=64,
        expert_dim=32,
        num_experts=8,
        tau=0.05
    )

    # Init parameters
    params = block.init(key, h)

    # Apply block
    out = block.apply(params, h)

    print("Output shape:", out.shape)
    print("Output:", out)

if __name__ == "__main__":
    main()
