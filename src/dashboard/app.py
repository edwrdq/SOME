from __future__ import annotations

from typing import Optional
import os
import time

import jax
import optax

from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import Footer, Header, Static

from .state import load_state, compute_expert_usage, TrainingState, append_sample_step, save_state, empty_state, append_real_step
from .views import HeatmapView, ExpertUsageView, TokenInspectorView
from src.model.model import CopyModel
from src.train.dataset import generate_copy_batch
from src.train.train_loop import train_step, eval_step


class SomeDashboard(App):
    CSS_PATH = "dashboard.tcss"
    BINDINGS = [
        ("left", "prev_step", "Previous step"),
        ("right", "next_step", "Next step"),
        ("up", "prev_token", "Previous token"),
        ("down", "next_token", "Next token"),
        ("p", "toggle_pause", "Pause/Resume"),
        ("e", "eval_now", "Eval"),
        ("s", "save_state", "Save"),
        ("q", "quit", "Quit"),
    ]

    def __init__(self, state_path: Optional[str] = None) -> None:
        super().__init__()
        # Initial config (fresh state each run, no carry-over)
        self._vocab = int(os.environ.get("SOME_VOCAB", "20"))
        self._seq_len = int(os.environ.get("SOME_SEQ_LEN", "8"))
        self._batch = int(os.environ.get("SOME_BATCH", "16"))
        self._hidden = int(os.environ.get("SOME_HIDDEN", "64"))
        self._expert_dim = int(os.environ.get("SOME_EXPERT_DIM", "32"))
        self._experts = int(os.environ.get("SOME_EXPERTS", "8"))
        self._lr = float(os.environ.get("SOME_LR", "1e-3"))

        self._state: TrainingState = empty_state(self._seq_len, self._experts)
        self._usage = compute_expert_usage(self._state)
        self._step_index = 0
        self._token_index = 0

        self.heatmap = HeatmapView(id="heatmap")
        self.usage = ExpertUsageView(id="usage")
        self.inspector = TokenInspectorView(id="inspector")

        # Training runtime (start paused until user toggles)
        self._paused = True

        self._model = CopyModel(self._vocab, self._hidden, self._expert_dim, self._experts)
        self._key = jax.random.PRNGKey(0)
        dummy = jax.random.randint(self._key, (self._batch, self._seq_len), 0, self._vocab)
        self._params = self._model.init(self._key, dummy)
        self._optimizer = optax.adam(self._lr)
        self._opt_state = self._optimizer.init(self._params)
        self._last_val = (float('nan'), 0.0)

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Vertical(id="root"):
            with Horizontal(id="top"):
                yield self.heatmap
                yield self.usage
            yield self.inspector
        yield Footer()

    def on_mount(self) -> None:
        self._refresh_views()
        # Drive training loop (does nothing while paused)
        self.set_interval(0.05, self._tick)

    def _refresh_views(self) -> None:
        if self._state.num_steps() == 0:
            # Show placeholders until training produces a step
            self.heatmap.update("No steps yet. Press 'p' to start training.")
            self.usage.set_usage([0 for _ in range(self._state.num_experts)])
            self.inspector.update("No data yet. Use arrow keys after training starts.")
            return
        self._step_index = max(0, min(self._step_index, self._state.num_steps() - 1))
        self._token_index = max(0, min(self._token_index, self._state.seq_len - 1))
        self.heatmap.set_data(self._state, self._step_index)
        self.usage.set_usage(self._usage)
        self.inspector.set_token(self._state, self._step_index, self._token_index)

    def action_prev_step(self) -> None:
        self._step_index = max(0, self._step_index - 1)
        self._refresh_views()

    def action_next_step(self) -> None:
        self._step_index = min(self._state.num_steps() - 1, self._step_index + 1)
        self._refresh_views()

    def action_prev_token(self) -> None:
        self._token_index = max(0, self._token_index - 1)
        self._refresh_views()

    def action_next_token(self) -> None:
        self._token_index = min(self._state.seq_len - 1, self._token_index + 1)
        self._refresh_views()

    def action_toggle_pause(self) -> None:
        self._paused = not self._paused

    def action_eval_now(self) -> None:
        # eval on a fixed batch
        key = jax.random.PRNGKey(42)
        _, inputs, targets = generate_copy_batch(key, self._batch, self._seq_len, self._vocab)
        loss, acc = eval_step(self._model, self._params, (inputs, targets))
        self._last_val = (float(loss), float(acc))
        # Trigger a view refresh; inspector shows current token info
        self._refresh_views()

    def action_save_state(self) -> None:
        path = os.environ.get("SOME_DASH_SAVE", "dashboard_state.json")
        save_state(self._state, path)

    # No-op status appender; views are fully recomputed on refresh

    def _tick(self) -> None:
        if self._paused:
            return
        # One train step
        self._key, inputs, targets = generate_copy_batch(self._key, self._batch, self._seq_len, self._vocab)

        # Capture real router weights for the current batch (before updating params)
        _, inter = self._model.apply(self._params, inputs, mutable=['intermediates'])
        weights_val = inter.get('intermediates', {}).get('router_weights') if isinstance(inter, dict) else None
        grid = None
        if weights_val is not None:
            import jax.numpy as jnp
            w = weights_val
            # Flatten possible nested lists to array
            try:
                w_arr = jnp.array(w)
            except Exception:
                w_arr = jnp.stack([jnp.asarray(x) for x in w], axis=0)
            # Expect shape (T, B, E) or (B, T, E)
            if w_arr.ndim == 3:
                T_candidate = w_arr.shape[0]
                if T_candidate == self._seq_len:
                    # (T, B, E)
                    grid_arr = w_arr.mean(axis=1)
                else:
                    # (B, T, E)
                    grid_arr = w_arr.mean(axis=0)
                grid = [[float(v) for v in row] for row in list(grid_arr)]

        # Compute update
        self._params, self._opt_state, loss = train_step(
            self._model, self._params, self._opt_state, self._optimizer, self._key, (inputs, targets)
        )

        # Append real step if captured; otherwise synthesize
        if grid is not None:
            append_real_step(self._state, grid, float(loss))
        else:
            append_sample_step(self._state, float(loss))
        self._usage = compute_expert_usage(self._state)
        self._step_index = self._state.num_steps() - 1
        self._refresh_views()


def main() -> None:
    app = SomeDashboard()
    app.run()


if __name__ == "__main__":
    main()
