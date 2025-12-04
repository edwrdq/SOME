import argparse
import curses
import time
from dataclasses import dataclass, asdict
from typing import List, Optional, Tuple

import jax
import jax.numpy as jnp
import optax

from model.model import CopyModel
from train.dataset import generate_copy_batch
from train.train_loop import train_step, eval_step
from utils.checkpoint import save_checkpoint


@dataclass
class ModelConfig:
    vocab: int = 20
    seq_len: int = 8
    batch: int = 16
    hidden: int = 64
    expert_dim: int = 32
    experts: int = 8
    lr: float = 1e-3
    seed: int = 0
    val_seed: int = 42


@dataclass
class TrainConfig:
    steps: int = 500
    eval_every: int = 20
    target_acc: float = 0.995
    patience: int = 3


@dataclass
class TUIState:
    model: CopyModel
    params: any
    opt_state: any
    optimizer: optax.GradientTransformation
    key: any
    val_inputs: jnp.ndarray
    val_targets: jnp.ndarray
    run: bool = True
    paused: bool = False
    step: int = 0
    train_loss: float = float("nan")
    val_loss: float = float("nan")
    val_acc: float = 0.0
    good: int = 0
    last_losses: List[float] = None
    last_preds: Optional[List[int]] = None
    last_input: Optional[List[int]] = None


def sparkline(data: List[float], width: int) -> str:
    if not data:
        return "".ljust(width)
    blocks = "▁▂▃▄▅▆▇█"
    n = len(data)
    if n > width:
        # downsample
        stride = n / width
        samples = [data[int(i * stride)] for i in range(width)]
    else:
        samples = list(data) + [data[-1]] * (width - n)
    lo = min(samples)
    hi = max(samples)
    if hi - lo < 1e-8:
        return (blocks[0] * width)
    out = []
    for v in samples:
        idx = int((v - lo) / (hi - lo) * (len(blocks) - 1))
        out.append(blocks[idx])
    return "".join(out)


def bar(value: float, width: int) -> str:
    value = max(0.0, min(1.0, value))
    filled = int(value * width)
    return ("█" * filled) + (" " * (width - filled))


def build_model(cfg: ModelConfig) -> CopyModel:
    return CopyModel(cfg.vocab, cfg.hidden, cfg.expert_dim, cfg.experts)


def init_state(mcfg: ModelConfig, tcfg: TrainConfig) -> TUIState:
    model = build_model(mcfg)
    key = jax.random.PRNGKey(mcfg.seed)
    dummy = jax.random.randint(key, (mcfg.batch, mcfg.seq_len), 0, mcfg.vocab)
    params = model.init(key, dummy)
    optimizer = optax.adam(mcfg.lr)
    opt_state = optimizer.init(params)
    val_key = jax.random.PRNGKey(mcfg.val_seed)
    _, val_inputs, val_targets = generate_copy_batch(val_key, mcfg.batch, mcfg.seq_len, mcfg.vocab)
    return TUIState(
        model=model,
        params=params,
        opt_state=opt_state,
        optimizer=optimizer,
        key=key,
        val_inputs=val_inputs,
        val_targets=val_targets,
        run=True,
        paused=False,
        step=0,
        train_loss=float("nan"),
        val_loss=float("nan"),
        val_acc=0.0,
        good=0,
        last_losses=[],
    )


def draw(stdscr, mcfg: ModelConfig, tcfg: TrainConfig, st: TUIState):
    stdscr.erase()
    h, w = stdscr.getmaxyx()
    title = f"SOME Tiny Transformer — RUNNING" if not st.paused else f"SOME Tiny Transformer — PAUSED"
    stdscr.addstr(0, 0, title[:w])
    stdscr.addstr(1, 0, (f"vocab={mcfg.vocab} seq={mcfg.seq_len} batch={mcfg.batch} hidden={mcfg.hidden} "
                         f"expert_dim={mcfg.expert_dim} experts={mcfg.experts} lr={mcfg.lr}")[:w])

    # Progress and metrics
    y = 3
    stdscr.addstr(y, 0, f"step: {st.step}/{tcfg.steps}")
    y += 1
    stdscr.addstr(y, 0, f"train_loss: {st.train_loss:.4f}")
    y += 1
    stdscr.addstr(y, 0, f"val_loss:   {st.val_loss:.4f}   val_acc: {st.val_acc:.4f}")
    y += 1
    stdscr.addstr(y, 0, f"target_acc: {tcfg.target_acc:.3f}  patience: {tcfg.patience}  good: {st.good}")
    y += 1

    # Acc bar
    accw = max(10, min(w - 10, 50))
    stdscr.addstr(y, 0, "acc: [" + bar(st.val_acc, accw) + "]")
    y += 2

    # Loss sparkline
    slw = min(w - 2, 80)
    stdscr.addstr(y, 0, "loss: " + sparkline(st.last_losses[-slw:], slw))
    y += 2

    # Predictions
    if st.last_input is not None and st.last_preds is not None:
        stdscr.addstr(y, 0, "last input:  " + " ".join(map(str, st.last_input))[:w])
        y += 1
        stdscr.addstr(y, 0, "last preds:  " + " ".join(map(str, st.last_preds))[:w])
        y += 1

    # Help
    y = h - 4
    help1 = "[q] quit  [p] pause/resume  [v] eval now  [s] save  [g] gen+predict  [i] input+predict  [c] config"
    stdscr.addstr(y, 0, help1[:w])
    y += 1
    help2 = "Arrows/keys while training refresh UI; early-stops on threshold"
    stdscr.addstr(y, 0, help2[:w])
    stdscr.refresh()


def do_eval(st: TUIState) -> None:
    loss, acc = eval_step(st.model, st.params, (st.val_inputs, st.val_targets))
    st.val_loss = float(loss)
    st.val_acc = float(acc)


def do_predict_random(mcfg: ModelConfig, st: TUIState) -> None:
    key = jax.random.PRNGKey(int(time.time()) & 0xFFFF)
    inputs = jax.random.randint(key, (1, mcfg.seq_len), 0, mcfg.vocab)
    logits = st.model.apply(st.params, inputs)
    preds = jnp.argmax(logits, axis=-1)
    st.last_input = list(map(int, inputs.tolist()[0]))
    st.last_preds = list(map(int, preds.tolist()[0]))


def do_predict_tokens(tokens_str: str, mcfg: ModelConfig, st: TUIState) -> Optional[str]:
    try:
        toks = [int(t) for t in tokens_str.strip().split() if t.strip()]
    except ValueError:
        return "Tokens must be integers"
    if len(toks) != mcfg.seq_len:
        return f"Need exactly {mcfg.seq_len} tokens"
    inputs = jnp.array([toks])
    logits = st.model.apply(st.params, inputs)
    preds = jnp.argmax(logits, axis=-1)
    st.last_input = toks
    st.last_preds = list(map(int, preds.tolist()[0]))
    return None


def input_line(stdscr, prompt: str) -> str:
    curses.echo()
    h, w = stdscr.getmaxyx()
    stdscr.addstr(h - 1, 0, (prompt + " ").ljust(w - 1))
    stdscr.refresh()
    try:
        s = stdscr.getstr(h - 1, len(prompt) + 1, w - len(prompt) - 2)
    except Exception:
        s = b""
    curses.noecho()
    return s.decode("utf-8", errors="ignore")


def _draw_menu(stdscr, title: str, rows: List[str], sel: int) -> None:
    stdscr.erase()
    h, w = stdscr.getmaxyx()
    stdscr.addstr(0, 0, title[:w])
    for i, line in enumerate(rows):
        prefix = "> " if i == sel else "  "
        stdscr.addstr(i + 2, 0, (prefix + line)[:w])
    stdscr.addstr(h - 2, 0, "[Enter] edit  [←/→ +/-] adjust  [Tab] next  [a] apply  [esc/q] cancel"[:w])
    stdscr.refresh()


def _format_cfg(mcfg: ModelConfig, tcfg: TrainConfig) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
    model_items = [
        ("vocab", str(mcfg.vocab)),
        ("seq_len", str(mcfg.seq_len)),
        ("batch", str(mcfg.batch)),
        ("hidden", str(mcfg.hidden)),
        ("expert_dim", str(mcfg.expert_dim)),
        ("experts", str(mcfg.experts)),
        ("lr", f"{mcfg.lr:g}"),
        ("seed", str(mcfg.seed)),
        ("val_seed", str(mcfg.val_seed)),
    ]
    train_items = [
        ("steps", str(tcfg.steps)),
        ("eval_every", str(tcfg.eval_every)),
        ("target_acc", f"{tcfg.target_acc:g}"),
        ("patience", str(tcfg.patience)),
    ]
    return model_items, train_items


def _apply_increment(name: str, value):
    # Return (dec, inc) functions
    if isinstance(value, int):
        step = 1
        def dec(v):
            return max(1, v - step)
        def inc(v):
            return v + step
        return dec, inc
    else:
        # float
        if name == "lr":
            def dec(v):
                return max(1e-6, v * 0.5)
            def inc(v):
                return min(1.0, v * 2.0)
        elif name == "target_acc":
            def dec(v):
                return max(0.0, v - 0.005)
            def inc(v):
                return min(1.0, v + 0.005)
        else:
            def dec(v):
                return v - 1.0
            def inc(v):
                return v + 1.0
        return dec, inc


def edit_config(stdscr, mcfg: ModelConfig, tcfg: TrainConfig) -> Tuple[ModelConfig, TrainConfig, bool]:
    # Returns possibly-updated configs and a boolean applied flag
    sel = 0
    fields = [
        ("vocab", "model"),
        ("seq_len", "model"),
        ("batch", "model"),
        ("hidden", "model"),
        ("expert_dim", "model"),
        ("experts", "model"),
        ("lr", "model"),
        ("seed", "model"),
        ("val_seed", "model"),
        ("steps", "train"),
        ("eval_every", "train"),
        ("target_acc", "train"),
        ("patience", "train"),
    ]
    while True:
        m_items, t_items = _format_cfg(mcfg, tcfg)
        rows = ["Model config:"] + [f"  {k}: {v}" for k, v in m_items] + ["Train config:"] + [f"  {k}: {v}" for k, v in t_items]
        # Map selection index to field index
        # Indexing: 0 title, 1..len(m_items) model, model_end+1 title, rest train
        menu_indices = list(range(len(rows)))
        # Default selection mapping: skip title rows
        selectable = [i for i, r in enumerate(rows) if not r.endswith(":")]
        sel = selectable[0] if sel not in selectable else sel
        _draw_menu(stdscr, "Config Editor", rows, sel)
        ch = stdscr.getch()
        if ch in (27, ord('q'), ord('Q')):  # esc or q
            return mcfg, tcfg, False
        elif ch in (curses.KEY_DOWN, ord('j')):
            # move to next selectable
            idx = selectable.index(sel)
            sel = selectable[min(len(selectable) - 1, idx + 1)]
        elif ch in (curses.KEY_UP, ord('k')):
            idx = selectable.index(sel)
            sel = selectable[max(0, idx - 1)]
        elif ch in (curses.KEY_RIGHT, ord('+')):
            name, scope = fields[selectable.index(sel)]
            if scope == "model":
                cur = getattr(mcfg, name if name != "seq_len" else "seq_len")
            else:
                cur = getattr(tcfg, name)
            dec, inc = _apply_increment(name, cur)
            newv = inc(cur)
            if scope == "model":
                setattr(mcfg, name, type(cur)(newv))
            else:
                setattr(tcfg, name, type(cur)(newv))
        elif ch in (curses.KEY_LEFT, ord('-')):
            name, scope = fields[selectable.index(sel)]
            cur = getattr(mcfg, name) if scope == "model" else getattr(tcfg, name)
            dec, inc = _apply_increment(name, cur)
            newv = dec(cur)
            if scope == "model":
                setattr(mcfg, name, type(cur)(newv))
            else:
                setattr(tcfg, name, type(cur)(newv))
        elif ch in (curses.KEY_ENTER, 10, 13, ord('e')):
            name, scope = fields[selectable.index(sel)]
            prompt = f"set {name} ="
            s = input_line(stdscr, prompt)
            try:
                if isinstance(getattr(mcfg if scope == "model" else tcfg, name), int):
                    v = int(s)
                else:
                    v = float(s)
            except Exception:
                continue
            if scope == "model":
                setattr(mcfg, name, v)
            else:
                setattr(tcfg, name, v)
        elif ch in (9,):  # tab
            idx = selectable.index(sel)
            sel = selectable[(idx + 1) % len(selectable)]
        elif ch in (ord('a'), ord('A')):
            return mcfg, tcfg, True



def run_tui(stdscr, mcfg: ModelConfig, tcfg: TrainConfig):
    curses.curs_set(0)
    stdscr.nodelay(True)
    stdscr.timeout(100)

    st = init_state(mcfg, tcfg)
    do_eval(st)

    last_eval = 0
    while st.run:
        # Train one step if not paused and steps remain
        if not st.paused and st.step < tcfg.steps:
            st.key, inputs, targets = generate_copy_batch(st.key, mcfg.batch, mcfg.seq_len, mcfg.vocab)
            st.params, st.opt_state, loss = train_step(
                st.model, st.params, st.opt_state, st.optimizer, st.key, (inputs, targets)
            )
            st.train_loss = float(loss)
            st.last_losses.append(st.train_loss)
            st.step += 1

            if st.step % tcfg.eval_every == 0:
                do_eval(st)
                if st.val_acc >= tcfg.target_acc:
                    st.good += 1
                else:
                    st.good = 0
                if st.good >= tcfg.patience:
                    st.paused = True

        draw(stdscr, mcfg, tcfg, st)

        # Handle input
        ch = stdscr.getch()
        if ch == -1:
            continue
        if ch in (ord('q'), ord('Q')):
            st.run = False
        elif ch in (ord('p'), ord('P'), ord(' ')):
            st.paused = not st.paused
        elif ch in (ord('v'), ord('V')):
            do_eval(st)
        elif ch in (ord('s'), ord('S')):
            ts = time.strftime("%Y%m%d-%H%M%S")
            base = f"checkpoints/tui-{ts}"
            cfg = {**asdict(mcfg), **asdict(tcfg)}
            save_checkpoint(base, st.params, cfg)
        elif ch in (ord('g'), ord('G')):
            do_predict_random(mcfg, st)
        elif ch in (ord('i'), ord('I')):
            s = input_line(stdscr, f"tokens (space-separated, {mcfg.seq_len} long)")
            err = do_predict_tokens(s, mcfg, st)
            if err:
                # show error briefly
                h, w = stdscr.getmaxyx()
                stdscr.addstr(h - 2, 0, ("Error: " + err).ljust(w - 1))
                stdscr.refresh()
                time.sleep(1.0)
        elif ch in (ord('c'), ord('C')):
            # Open config editor; rebuild model/state if applied
            new_mcfg = ModelConfig(**asdict(mcfg))
            new_tcfg = TrainConfig(**asdict(tcfg))
            new_mcfg, new_tcfg, applied = edit_config(stdscr, new_mcfg, new_tcfg)
            if applied:
                mcfg = new_mcfg
                tcfg = new_tcfg
                st = init_state(mcfg, tcfg)
                do_eval(st)


def parse_args():
    p = argparse.ArgumentParser(description="TUI for SOME Tiny Transformer")
    p.add_argument("--vocab", type=int, default=20)
    p.add_argument("--seq-len", type=int, default=8)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--expert-dim", type=int, default=32)
    p.add_argument("--experts", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--val-seed", type=int, default=42)

    p.add_argument("--steps", type=int, default=500)
    p.add_argument("--eval-every", type=int, default=20)
    p.add_argument("--target-acc", type=float, default=0.995)
    p.add_argument("--patience", type=int, default=3)
    return p.parse_args()


def main():
    args = parse_args()
    mcfg = ModelConfig(
        vocab=args.vocab,
        seq_len=args.seq_len,
        batch=args.batch,
        hidden=args.hidden,
        expert_dim=args.expert_dim,
        experts=args.experts,
        lr=args.lr,
        seed=args.seed,
        val_seed=args.val_seed,
    )
    tcfg = TrainConfig(
        steps=args.steps,
        eval_every=args.eval_every,
        target_acc=args.target_acc,
        patience=args.patience,
    )
    curses.wrapper(run_tui, mcfg, tcfg)


if __name__ == "__main__":
    main()
