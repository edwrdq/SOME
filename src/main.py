# src/main.py
import jax
import optax
from model.model import CopyModel
from train.dataset import generate_copy_batch
from train.train_loop import train_step, eval_step

def main():
    vocab = 20
    seq_len = 8
    batch = 16
    hidden = 64
    expert_dim = 32
    experts = 8

    model = CopyModel(vocab, hidden, expert_dim, experts)

    key = jax.random.PRNGKey(0)
    dummy = jax.random.randint(key, (batch, seq_len), 0, vocab)
    params = model.init(key, dummy)

    global optimizer
    optimizer = optax.adam(1e-3)
    opt_state = optimizer.init(params)

    # Early stopping config
    target_acc = 0.995
    eval_every = 20
    patience = 3
    good = 0

    # Fixed validation batch
    val_key = jax.random.PRNGKey(42)
    _, val_inputs, val_targets = generate_copy_batch(val_key, batch, seq_len, vocab)

    for step in range(500):
        key, inp, tgt = generate_copy_batch(key, batch, seq_len, vocab)
        params, opt_state, loss = train_step(
            model, params, opt_state, optimizer, key, (inp, tgt)
        )

        if step % eval_every == 0:
            val_loss, val_acc = eval_step(model, params, (val_inputs, val_targets))
            print(f"step {step}: train_loss={float(loss):.4f} val_loss={float(val_loss):.4f} val_acc={float(val_acc):.4f}")
            if float(val_acc) >= target_acc:
                good += 1
            else:
                good = 0
            if good >= patience:
                print(f"Early stop at step {step}: val_acc {float(val_acc):.4f} >= {target_acc} for {patience} evals")
                break

if __name__ == "__main__":
    main()
