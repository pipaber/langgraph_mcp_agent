import os
import yaml
from functools import lru_cache


@lru_cache(maxsize=1)
def load_config(path: str | None = None) -> dict:
    """Load config.yaml from repository root or provided path.

    Defaults to project root `config.yaml`.
    """
    if path is None:
        # Try to find config.yaml relative to CWD and repo layout
        candidates = [
            os.path.join(os.getcwd(), "config.yaml"),
            os.path.join(os.path.dirname(os.getcwd()), "config.yaml"),
        ]
        for p in candidates:
            if os.path.exists(p):
                path = p
                break
    if not path:
        raise FileNotFoundError("config.yaml not found")
    with open(path, "r") as f:
        return yaml.safe_load(f)
