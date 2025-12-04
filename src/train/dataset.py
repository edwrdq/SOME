# src/train/dataset.py
import jax.numpy as jnp
import jax

def generate_copy_batch(key, batch_size, seq_len, vocab_size):
    """
    Generates a batch of random sequences and their identical targets.
    """
    key, subkey = jax.random.split(key)
    inputs = jax.random.randint(subkey, (batch_size, seq_len), 0, vocab_size)
    
    # Targets are identical to inputs
    targets = inputs.copy()

    return key, inputs, targets
