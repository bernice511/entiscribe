import json
import subprocess
from typing import Any, TypeVar

from pydantic import BaseModel

ModelT = TypeVar("ModelT", bound=BaseModel)


class ClaudeCLIError(RuntimeError):
    pass


def run_claude(
    prompt: str,
    *,
    json_schema: dict[str, Any] | None = None,
    model: str | None = None,
    timeout: int = 120,
) -> str | dict[str, Any]:
    command = ["claude", "-p", prompt, "--output-format", "json"]
    if json_schema is not None:
        command += ["--json-schema", json.dumps(json_schema)]
    if model is not None:
        command += ["--model", model]

    try:
        completed = subprocess.run(command, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        raise ClaudeCLIError(f"claude CLI timed out after {timeout}s") from exc
    except FileNotFoundError as exc:
        raise ClaudeCLIError(
            "claude CLI not found on PATH. Make sure `claude` is installed and on PATH for "
            "whichever shell/process launched this app (run `which claude` there to check)."
        ) from exc

    if completed.returncode != 0:
        raise ClaudeCLIError(f"claude CLI exited {completed.returncode}: {completed.stderr.strip()}")

    try:
        envelope = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise ClaudeCLIError(f"claude CLI returned non-JSON output: {completed.stdout[:500]!r}") from exc

    if envelope.get("is_error"):
        raise ClaudeCLIError(f"claude CLI reported an error: {envelope.get('result')}")

    if json_schema is not None:
        return envelope["structured_output"]
    return envelope["result"]


def run_claude_structured(
    prompt: str,
    schema_model: type[ModelT],
    *,
    model: str | None = None,
    timeout: int = 120,
) -> ModelT:
    data = run_claude(prompt, json_schema=schema_model.model_json_schema(), model=model, timeout=timeout)
    return schema_model.model_validate(data)
