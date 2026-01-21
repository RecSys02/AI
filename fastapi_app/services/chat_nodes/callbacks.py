from typing import List


def build_callbacks_config(callbacks: List[object] | None) -> dict | None:
    """Return a LangChain config dict when callbacks are available."""
    if callbacks:
        return {"callbacks": callbacks}
    return None
