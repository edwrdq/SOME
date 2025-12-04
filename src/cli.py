import argparse
import sys
from dataclasses import asdict, dataclass

import jax
import jax.numpy as jnp
import optax

from model.model import CopyModel
from train.dataset import generate_copy_batch
from train.train_loop import train_step, eval_step
from utils.checkpoint import save_checkpoint, load_checkpoint


@dataclass
class ModelConfig:
    vocab_size: int
    hidden_dim: int
    expert_dim: int
    num_experts: int


def build_model(cfg: ModelConfig) -> CopyModel:
    return CopyModel(
        vocab_size=cfg.vocab_size,
        hidden_dim=cfg.hidden_dim,
        expert_dim=cfg.expert_dim,
        num_experts=cfg.num_experts,
    )


def cmd_train(args: argparse.Namespace) -> None:
    cfg = ModelConfig(
        vocab_size=args.vocab,
        hidden_dim=args.hidden,
        expert_dim=args.expert_dim,
        num_experts=args.experts,
    )
    model = build_model(cfg)

    key = jax.random.PRNGKey(args.seed)
    dummy = jax.random.randint(key, (args.batch, args.seq_len), 0, args.vocab)
    params = model.init(key, dummy)

    optimizer = optax.adam(args.lr)
    opt_state = optimizer.init(params)

    # Fixed validation batch
    val_key = jax.random.PRNGKey(args.val_seed)
    _, val_inputs, val_targets = generate_copy_batch(val_key, args.batch, args.seq_len, args.vocab)

    good = 0
    for step in range(args.steps):
        key, inputs, targets = generate_copy_batch(key, args.batch, args.seq_len, args.vocab)
        params, opt_state, loss = train_step(
            model, params, opt_state, optimizer, key, (inputs, targets)
        )

        if step % args.eval_every == 0:
            val_loss, val_acc = eval_step(model, params, (val_inputs, val_targets))
            print(
                f"step {step}: train_loss={float(loss):.4f} val_loss={float(val_loss):.4f} val_acc={float(val_acc):.4f}",
                flush=True,
            )
            if float(val_acc) >= args.target_acc:
                good += 1
            else:
                good = 0
            if good >= args.patience:
                print(
                    f"Early stop at step {step}: val_acc {float(val_acc):.4f} >= {args.target_acc} for {args.patience} evals",
                    flush=True,
                )
                break

    if args.save:
        base = save_checkpoint(args.save, params, {**asdict(cfg), "seq_len": args.seq_len})
        print(f"Saved checkpoint to {base}.msgpack and {base}.json")


def _init_like_for_load(model: CopyModel, batch: int, seq_len: int, vocab: int, seed: int):
    key = jax.random.PRNGKey(seed)
    dummy = jax.random.randint(key, (batch, seq_len), 0, vocab)
    return model.init(key, dummy)


def cmd_eval(args: argparse.Namespace) -> None:
    cfg = ModelConfig(
        vocab_size=args.vocab,
        hidden_dim=args.hidden,
        expert_dim=args.expert_dim,
        num_experts=args.experts,
    )
    model = build_model(cfg)
    params_like = _init_like_for_load(model, args.batch, args.seq_len, args.vocab, args.seed)
    params, meta = load_checkpoint(args.load, params_like)

    total_acc = 0.0
    total_loss = 0.0
    key = jax.random.PRNGKey(args.seed)
    for _ in range(args.eval_steps):
        key, inputs, targets = generate_copy_batch(key, args.batch, args.seq_len, args.vocab)
        loss, acc = eval_step(model, params, (inputs, targets))
        total_acc += float(acc)
        total_loss += float(loss)
    print(
        f"eval_mean_loss={(total_loss/args.eval_steps):.4f} eval_mean_acc={(total_acc/args.eval_steps):.4f}",
        flush=True,
    )


def cmd_predict(args: argparse.Namespace) -> None:
    cfg = ModelConfig(
        vocab_size=args.vocab,
        hidden_dim=args.hidden,
        expert_dim=args.expert_dim,
        num_experts=args.experts,
    )
    model = build_model(cfg)
    params_like = _init_like_for_load(model, 1, args.seq_len, args.vocab, args.seed)
    params, meta = load_checkpoint(args.load, params_like)

    if args.input:
        toks = [int(t) for t in args.input.strip().split()]  # space-separated ints
        if len(toks) != args.seq_len:
            print(f"--input must have exactly seq_len={args.seq_len} tokens", file=sys.stderr)
            sys.exit(2)
        inputs = jnp.array([toks])
    else:
        key = jax.random.PRNGKey(args.seed)
        inputs = jax.random.randint(key, (1, args.seq_len), 0, args.vocab)

    logits = model.apply(params, inputs)
    preds = jnp.argmax(logits, axis=-1)
    print("input:", inputs.tolist()[0])
    print("preds:", preds.tolist()[0])


def make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="some-cli", description="Tiny transformer SOME demo")
    sub = p.add_subparsers(dest="cmd", required=True)

    def add_model_args(sp):
        sp.add_argument("--vocab", type=int, default=20)
        sp.add_argument("--seq-len", type=int, default=8)
        sp.add_argument("--batch", type=int, default=16)
        sp.add_argument("--hidden", type=int, default=64)
        sp.add_argument("--expert-dim", type=int, default=32)
        sp.add_argument("--experts", type=int, default=8)
        sp.add_argument("--seed", type=int, default=0)

    # train
    sp_train = sub.add_parser("train", help="Train on copy task with early stopping")
    add_model_args(sp_train)
    sp_train.add_argument("--steps", type=int, default=500)
    sp_train.add_argument("--lr", type=float, default=1e-3)
    sp_train.add_argument("--eval-every", type=int, default=20)
    sp_train.add_argument("--target-acc", type=float, default=0.995)
    sp_train.add_argument("--patience", type=int, default=3)
    sp_train.add_argument("--val-seed", type=int, default=42)
    sp_train.add_argument("--save", type=str, default="", help="Base path to save checkpoint (without extension)")
    sp_train.set_defaults(func=cmd_train)

    # eval
    sp_eval = sub.add_parser("eval", help="Evaluate a saved checkpoint on the copy task")
    add_model_args(sp_eval)
    sp_eval.add_argument("--load", type=str, required=True, help="Checkpoint base path (with or without extension)")
    sp_eval.add_argument("--eval-steps", type=int, default=20)
    sp_eval.set_defaults(func=cmd_eval)

    # predict
    sp_pred = sub.add_parser("predict", help="Predict tokens for an input sequence")
    add_model_args(sp_pred)
    sp_pred.add_argument("--load", type=str, required=True, help="Checkpoint base path (with or without extension)")
    sp_pred.add_argument("--input", type=str, default="", help="Space-separated token ids; if omitted, uses random input")
    sp_pred.set_defaults(func=cmd_predict)

    return p


def main(argv=None):
    parser = make_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()

