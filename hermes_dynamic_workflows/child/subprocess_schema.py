"""Prompt-and-validate structured output for runners with no native schema flag.

The Hermes runner constrains a child's final value with a real tool call
(`child/structured_output.py`). Subprocess runners that expose no equivalent —
pi has neither `--json-schema` nor a schema-constrained `--output-format`; its
`--mode json` is an *event stream*, not constrained output — get the same
guarantee the only way left: ask for JSON, parse it, validate it against the
same schema, and re-prompt with the concrete validation errors on a mismatch.

Retry budget is shared with the native path (`MAX_STRUCTURED_OUTPUT_RETRIES`)
so a schema failure costs the same either way regardless of runner.
"""

from __future__ import annotations

import json
import re
from typing import Any, Callable

from .structured_output import MAX_STRUCTURED_OUTPUT_RETRIES, _validation_errors
from ..core.errors import ChildAgentError

_FENCED_JSON = re.compile(r"```(?:json)?\s*\n(.*?)\n?```", re.DOTALL)


def build_schema_prompt(prompt: str, schema: dict[str, Any]) -> str:
    """Append the "return one JSON object matching this schema" contract."""
    rendered = json.dumps(schema, ensure_ascii=False, indent=2, sort_keys=True)
    return (
        f"{prompt}\n\n"
        "---\n"
        "Return your final answer as a SINGLE JSON object that validates "
        "against this JSON Schema:\n\n"
        f"```json\n{rendered}\n```\n\n"
        "Your last message must contain that JSON object inside one ```json "
        "fenced code block and nothing else after it. Do not add commentary "
        "inside the block."
    )


def build_retry_prompt(errors: list[str]) -> str:
    joined = "; ".join(errors) if errors else "the output was not valid JSON"
    return (
        "Your previous answer did not satisfy the required JSON Schema: "
        f"{joined}.\n\n"
        "Reply again with the corrected SINGLE JSON object inside one ```json "
        "fenced code block. Do not explain the fix."
    )


def extract_json_value(text: str) -> tuple[bool, Any]:
    """Pull the intended JSON value out of a free-text final message.

    Tries, in order: every fenced block (last one wins — models often show a
    draft before the final answer), then the whole message, then the widest
    balanced brace/bracket span. Returns (found, value).
    """
    candidates: list[str] = []
    candidates.extend(match.group(1) for match in _FENCED_JSON.finditer(text or ""))
    stripped = (text or "").strip()
    if stripped:
        candidates.append(stripped)
    span = _widest_bracket_span(stripped)
    if span:
        candidates.append(span)

    for candidate in reversed(candidates):
        try:
            return True, json.loads(candidate)
        except (TypeError, ValueError):
            continue
    return False, None


def _widest_bracket_span(text: str) -> str:
    starts = [index for index in (text.find("{"), text.find("[")) if index >= 0]
    if not starts:
        return ""
    start = min(starts)
    closer = "}" if text[start] == "{" else "]"
    end = text.rfind(closer)
    if end <= start:
        return ""
    return text[start : end + 1]


def run_with_schema(
    prompt: str,
    schema: dict[str, Any],
    invoke: Callable[[str, int], str],
) -> tuple[Any, int]:
    """Drive ``invoke`` until its output validates, or the retry budget runs out.

    ``invoke(prompt, attempt)`` runs one child turn and returns its final text.
    Returns ``(validated_value, attempts)``.
    """
    message = build_schema_prompt(prompt, schema)
    last_errors: list[str] = []
    for attempt in range(1, MAX_STRUCTURED_OUTPUT_RETRIES + 1):
        content = invoke(message, attempt)
        found, value = extract_json_value(content)
        if not found:
            last_errors = ["root: response did not contain a JSON object"]
        else:
            last_errors = _validation_errors(value, schema)
            if not last_errors:
                return value, attempt
        message = build_retry_prompt(last_errors)
    raise ChildAgentError(
        "Failed to provide valid structured output after "
        f"{MAX_STRUCTURED_OUTPUT_RETRIES} attempts: {'; '.join(last_errors)}"
    )
