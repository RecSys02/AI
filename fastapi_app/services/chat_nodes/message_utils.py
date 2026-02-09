import json
from typing import Iterable, List, Tuple


def _coerce_content(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False)
    except Exception:
        return str(value)


def normalize_messages(messages: Iterable[Tuple[str, object]]) -> List[Tuple[str, str]]:
    """Collapse system messages and coerce content to strings."""
    system_parts: List[str] = []
    cleaned: List[Tuple[str, str]] = []
    for role, content in messages:
        role_value = str(role or "").strip().lower()
        if role_value == "system":
            text = _coerce_content(content).strip()
            if text:
                system_parts.append(text)
            continue
        if role_value not in {"user", "assistant", "tool"}:
            role_value = "user"
        cleaned.append((role_value, _coerce_content(content)))
    if system_parts:
        cleaned.insert(0, ("system", "\n\n".join(system_parts)))
    return cleaned
