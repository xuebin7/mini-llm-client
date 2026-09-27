"""JSON response formats and local Pydantic validation."""

import json
from typing import Any

from pydantic import BaseModel


def _make_strict(node: dict[str, Any]) -> None:
    """Close object schemas and require every field, including defaulted ones.

    Walk schema keywords only: property names and literal defaults are data.
    References remain intact; their definitions are processed separately.
    """
    node.pop("default", None)
    if node.get("type") == "object":
        if node.get("additionalProperties", False) is not False:
            raise ValueError(
                "Strict JSON Schema does not support open dictionaries or "
                "extra='allow'; use a model with named fields or JSON mode"
            )
        node["additionalProperties"] = False
        node["required"] = list(node.get("properties", {}))

    for keyword in ("properties", "$defs"):
        for child in node.get(keyword, {}).values():
            _make_strict(child)
    items = node.get("items")
    if isinstance(items, dict):
        _make_strict(items)
    for keyword in ("anyOf", "allOf", "oneOf", "prefixItems"):
        for child in node.get(keyword, []):
            _make_strict(child)


def build_response_format(schema: type[BaseModel]) -> dict[str, Any]:
    """Convert a Pydantic object model to a strict Chat Completions format.

    Providers support different JSON Schema subsets and validate remaining
    constraints themselves. This is not a universal JSON Schema compiler.
    """
    json_schema = schema.model_json_schema()
    root = json_schema
    if "$ref" in root:
        prefix = "#/$defs/"
        reference = root["$ref"]
        if reference.startswith(prefix):
            root = json_schema.get("$defs", {}).get(reference[len(prefix) :], {})
    if root.get("type") != "object" or "anyOf" in root:
        raise ValueError("Strict JSON Schema requires an object model at the root")
    _make_strict(json_schema)
    return {
        "type": "json_schema",
        "json_schema": {
            "name": schema.__name__,
            "schema": json_schema,
            "strict": True,
        },
    }


def parse_structured_output[T: BaseModel](content: str, schema: type[T]) -> T:
    """Parse JSON, then validate it using the model's Pydantic configuration.

    JSONDecodeError and ValidationError intentionally retain their distinct
    meanings. Pydantic's default coercion (e.g. "30" to 30) is preserved.
    """
    return schema.model_validate(json.loads(content))
