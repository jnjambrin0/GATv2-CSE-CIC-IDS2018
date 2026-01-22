import json
from typing import Any, Dict, Optional


def format_sse(event: str, data: Dict[str, Any], event_id: Optional[str] = None) -> str:
    payload = json.dumps(data, ensure_ascii=False)
    lines = []
    if event:
        lines.append(f"event: {event}")
    if event_id is not None:
        lines.append(f"id: {event_id}")
    for line in payload.splitlines():
        lines.append(f"data: {line}")
    lines.append("")
    return "\n".join(lines) + "\n"
