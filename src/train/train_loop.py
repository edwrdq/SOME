import jax
import jax.numpy as jnp
import optax


def cross_entropy_loss(logits, targets):
    onehot = jax.nn.one_hot(targets, logits.shape[-1])
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    loss = -jnp.sum(onehot * log_probs, axis=-1)
    return jnp.mean(loss)


def accuracy(logits, targets):
    preds = jnp.argmax(logits, axis=-1)
    return jnp.mean((preds == targets).astype(jnp.float32))


def eval_step(model, params, batch):
    inputs, targets = batch
    logits = model.apply(params, inputs)
    loss = cross_entropy_loss(logits, targets)
    acc = accuracy(logits, targets)
    return loss, acc

def train_step(model, params, opt_state, optimizer, key, batch):
    inputs, targets = batch

    def loss_fn(params):
        logits = model.apply(params, inputs)
        loss = cross_entropy_loss(logits, targets)
        return loss

    grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = grad_fn(params)
    updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)

    return params, opt_state, loss
