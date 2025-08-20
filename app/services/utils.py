from __future__ import annotations

from typing import Any

def get_nested_value(d: Any, path: str) -> Any:
    for part in path.split('.'):
        if isinstance(d, dict):
            d = d.get(part, "")
        else:
            return ""
    return d
