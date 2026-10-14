"""Small pure helpers shared by the app and unit tests."""

import json
from collections.abc import Iterable
from typing import Any


def parse_json_object(response: str) -> dict[str, Any]:
    text = response.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1]
        if text.endswith("```"):
            text = text[:-3].strip()
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < start:
        raise ValueError("The model response did not contain a JSON object.")
    try:
        result = json.loads(text[start : end + 1])
    except json.JSONDecodeError as exc:
        raise ValueError("The model returned invalid JSON. Please retry.") from exc
    if not isinstance(result, dict):
        raise ValueError("The model response must be a JSON object.")
    return result


def response_text(message: Any) -> str:
    content = getattr(message, "content", message)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            str(item.get("text", "")) if isinstance(item, dict) else str(item)
            for item in content
        )
    return str(content)


def source_references(documents: Iterable[Any]) -> list[str]:
    references: list[str] = []
    seen: set[tuple[str, str]] = set()
    for document in documents:
        metadata = getattr(document, "metadata", {})
        source = str(metadata.get("source", "Uploaded document"))
        page = metadata.get("page")
        page_label = str(page) if page is not None else "?"
        key = (source, page_label)
        if key not in seen:
            references.append(f"{source} · p. {page_label}")
            seen.add(key)
    return references