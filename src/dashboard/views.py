from __future__ import annotations

from typing import List

from textual.widget import Widget
from textual.widgets import Static

from .state import TrainingState, StepData


HEATMAP_SHADES = " ░▒▓█"


class HeatmapView(Static):
    """Renders a (seq_len x num_experts) grid for a selected step."""

    DEFAULT_CSS = """
    HeatmapView {
        overflow: auto;
        padding: 1 1;
        border: heavy #444444;
    }
    """

    def set_data(self, state: TrainingState, step_index: int) -> None:
        step = state.steps[step_index]
        content_lines: List[str] = []
        # Normalize over full grid for consistent shading
        grid = step.router_logits
        lo = min(min(r) for r in grid)
        hi = max(max(r) for r in grid)
        rng = hi - lo if hi > lo else 1.0

        for t, row in enumerate(grid):
            shades = []
            for v in row:
                x = (v - lo) / rng
                idx = min(len(HEATMAP_SHADES) - 1, int(x * (len(HEATMAP_SHADES) - 1)))
                shades.append(HEATMAP_SHADES[idx])
            content_lines.append(f"t{t:02d} " + "".join(shades))
        self.update("\n".join(content_lines))


class ExpertUsageView(Static):
    """Shows expert selection frequency over the whole history."""

    DEFAULT_CSS = """
    ExpertUsageView {
        overflow: auto;
        padding: 1 1;
        border: heavy #444444;
    }
    """

    def set_usage(self, counts: List[int]) -> None:
        total = sum(counts) or 1
        lines = ["Expert Usage (argmax freq):"]
        max_bar = 30
        for i, c in enumerate(counts):
            frac = c / total
            bar = "█" * int(frac * max_bar)
            lines.append(f"E{i:02d} {frac:6.2%} {bar}")
        self.update("\n".join(lines))


class TokenInspectorView(Static):
    """Shows logits vector, chosen experts, and loss for selected step+token."""

    DEFAULT_CSS = """
    TokenInspectorView {
        overflow: auto;
        padding: 1 1;
        border: heavy #444444;
    }
    """

    def set_token(self, state: TrainingState, step_index: int, token_index: int) -> None:
        step = state.steps[step_index]
        row = step.router_logits[token_index]
        # Top-3 experts
        top = sorted([(v, i) for i, v in enumerate(row)], reverse=True)[:3]
        logits_str = " ".join(f"{v:5.2f}" for v in row)
        chosen = ", ".join([f"E{i}={v:5.2f}" for v, i in top])
        lines = [
            f"Step {step_index}  Token {token_index}",
            f"Loss: {step.loss:.4f}",
            f"Logits: {logits_str}",
            f"Top-3: {chosen}",
        ]
        self.update("\n".join(lines))

