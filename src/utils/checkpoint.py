import os
import json
from typing import Any, Tuple

import flax.serialization as serialization


def save_checkpoint(path: str, params: Any, config: dict) -> str:
    """Save parameters (Flax PyTree) and a small JSON config next to it.

    Writes two files:
    - `<path>.msgpack` with serialized params
    - `<path>.json` with the config dict
    Returns the base path provided.
    """
    base, _ = os.path.splitext(path)
    params_path = base + ".msgpack"
    config_path = base + ".json"

    os.makedirs(os.path.dirname(params_path) or ".", exist_ok=True)
    data = serialization.to_bytes(params)
    with open(params_path, "wb") as f:
        f.write(data)
    with open(config_path, "w") as f:
        json.dump(config, f)
    return base


def load_checkpoint(path: str, params_like: Any) -> Tuple[Any, dict]:
    """Load parameters and config written by save_checkpoint.

    `params_like` should be a params PyTree with the same structure as the
    expected checkpoint (e.g., from model.init(...)) so from_bytes can map.
    Returns (params, config_dict).
    """
    base, _ = os.path.splitext(path)
    params_path = base + ".msgpack"
    config_path = base + ".json"

    with open(params_path, "rb") as f:
        raw = f.read()
    params = serialization.from_bytes(params_like, raw)

    config = {}
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            config = json.load(f)

    return params, config

