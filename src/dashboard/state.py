from __future__ import annotations

import json
import math
import os
import random
from dataclasses import dataclass, asdict
from typing import List, Optional, Tuple


@dataclass
class StepData:
    # router_logits[token][expert] -> float (pre-softmax or normalized)
    router_logits: List[List[float]]
    # scalar loss for the step
    loss: float


@dataclass
class TrainingState:
    steps: List[StepData]
    seq_len: int
    num_experts: int
    seed: int = 7
    base_bias: Optional[List[float]] = None

    def num_steps(self) -> int:
        return len(self.steps)

    def ensure_bias(self) -> None:
        if self.base_bias is None:
            rnd = random.Random(self.seed)
            self.base_bias = [rnd.uniform(-0.5, 0.5) for _ in range(self.num_experts)]


def _softmax(vals: List[float]) -> List[float]:
    m = max(vals)
    exps = [math.exp(v - m) for v in vals]
    s = sum(exps)
    return [e / s for e in exps]


def _normalize_grid(grid: List[List[float]]) -> List[List[float]]:
    # Normalize across the entire grid to [0,1]
    lo = min(min(row) for row in grid)
    hi = max(max(row) for row in grid)
    if hi - lo < 1e-12:
        return [[0.0 for _ in row] for row in grid]
    return [[(v - lo) / (hi - lo) for v in row] for row in grid]


def generate_sample_state(steps: int = 120, seq_len: int = 16, num_experts: int = 8, seed: int = 7) -> TrainingState:
    rnd = random.Random(seed)
    steps_data: List[StepData] = []
    # Create a slowly shifting preference over experts to make usage non-uniform
    base_bias = [rnd.uniform(-0.5, 0.5) for _ in range(num_experts)]
    for s in range(steps):
        router_logits: List[List[float]] = []
        phase = 2.0 * math.pi * (s / max(1, steps - 1))
        # token-specific router distributions
        for t in range(seq_len):
            # A wavy pattern across experts
            vals = []
            for e in range(num_experts):
                val = (
                    1.2 * math.sin(phase + e * 0.6 + t * 0.15)
                    + 0.8 * math.cos(phase * 0.5 + e * 0.3)
                    + base_bias[e]
                    + rnd.uniform(-0.15, 0.15)
                )
                vals.append(val)
            # Use softmax to get prob-like logits then keep as logits for visualization
            probs = _softmax(vals)
            router_logits.append(probs)
        # Make loss trend down with noise
        loss = max(0.05, 3.0 * math.exp(-s / max(1, steps / 6)) + rnd.uniform(-0.05, 0.05))
        steps_data.append(StepData(router_logits=router_logits, loss=loss))
    return TrainingState(steps=steps_data, seq_len=seq_len, num_experts=num_experts, seed=seed, base_bias=base_bias)


def append_sample_step(state: TrainingState, loss_value: float) -> None:
    """Append a synthetically generated router grid with provided loss value.

    Uses state.base_bias and a smooth phase progression based on current step count.
    """
    state.ensure_bias()
    s = state.num_steps()
    phase = 2.0 * math.pi * (s / max(1, s + 8))
    grid: List[List[float]] = []
    for t in range(state.seq_len):
        vals = []
        for e in range(state.num_experts):
            val = (
                1.2 * math.sin(phase + e * 0.6 + t * 0.15)
                + 0.8 * math.cos(phase * 0.5 + e * 0.3)
                + state.base_bias[e]
            )
            vals.append(val)
        probs = _softmax(vals)
        grid.append(probs)
    state.steps.append(StepData(router_logits=grid, loss=float(loss_value)))


def empty_state(seq_len: int, num_experts: int, seed: int = 7) -> TrainingState:
    rnd = random.Random(seed)
    base_bias = [rnd.uniform(-0.5, 0.5) for _ in range(num_experts)]
    return TrainingState(steps=[], seq_len=seq_len, num_experts=num_experts, seed=seed, base_bias=base_bias)


def append_real_step(state: TrainingState, grid: List[List[float]], loss_value: float) -> None:
    # Accepts a (seq_len x num_experts) grid of floats. Values need not be normalized.
    # We store directly; HeatmapView normalizes per-step when rendering.
    state.steps.append(StepData(router_logits=grid, loss=float(loss_value)))


def compute_expert_usage(state: TrainingState) -> List[int]:
    counts = [0 for _ in range(state.num_experts)]
    for step in state.steps:
        for token_row in step.router_logits:
            # choose argmax expert for token
            best = max(range(state.num_experts), key=lambda i: token_row[i])
            counts[best] += 1
    return counts


def _to_dict(state: TrainingState) -> dict:
    return {
        "seq_len": state.seq_len,
        "num_experts": state.num_experts,
        "seed": state.seed,
        "base_bias": state.base_bias,
        "steps": [
            {"router_logits": s.router_logits, "loss": s.loss}
            for s in state.steps
        ],
    }


def _from_dict(d: dict) -> TrainingState:
    steps = [StepData(router_logits=s["router_logits"], loss=float(s["loss"])) for s in d["steps"]]
    return TrainingState(
        steps=steps,
        seq_len=int(d["seq_len"]),
        num_experts=int(d["num_experts"]),
        seed=int(d.get("seed", 7)),
        base_bias=d.get("base_bias"),
    )


_CACHE: dict[str, TrainingState] = {}


def load_state(path: Optional[str] = None) -> TrainingState:
    """
    Load TrainingState from JSON file if `path` exists; otherwise generate a sample state.
    Results are cached by path key ("<generated>").
    """
    key = path or "<generated>"
    if key in _CACHE:
        return _CACHE[key]
    if path and os.path.exists(path):
        with open(path, "r") as f:
            data = json.load(f)
        state = _from_dict(data)
    else:
        state = generate_sample_state()
    _CACHE[key] = state
    return state


def save_state(state: TrainingState, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(_to_dict(state), f)
    _CACHE[path] = state
