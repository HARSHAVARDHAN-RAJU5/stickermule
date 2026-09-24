"""Stage 06, part three: the models that can word a message.

Each writer does one thing: take instructions and a prompt, return text. None
of them sees an image, a finding or a route, and nothing they return is used
until the guard has passed it.

Only the standard library is used for HTTP, so adding a provider does not add a
dependency. Keys come from the environment or a local .env file, never from
the repo.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

ROOT = Path(__file__).resolve().parents[2]


class WriterUnavailable(Exception):
    """The writer cannot be used: no key, no server, or the call failed."""


@dataclass(frozen=True)
class Reply:
    text: str
    tokens_in: int | None = None
    tokens_out: int | None = None


class Writer(Protocol):
    name: str

    def complete(self, system: str, prompt: str) -> Reply: ...


def env(key: str) -> str | None:
    """A setting from the environment, or from .env at the project root."""
    if os.environ.get(key):
        return os.environ[key]
    path = ROOT / ".env"
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            name, sep, value = line.partition("=")
            if sep and name.strip() == key:
                return value.strip().strip("'\"") or None
    return None


# Rate limited or briefly overloaded: worth one more try after a pause.
TRANSIENT = frozenset({429, 500, 502, 503, 504})


def _post(
    url: str, body: dict, headers: dict[str, str], timeout: float, wait_s: float = 10.0
) -> dict:
    request = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json", **headers},
        method="POST",
    )
    for last in (False, True):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            if exc.code in TRANSIENT and not last:
                time.sleep(wait_s)
                continue
            detail = " ".join(exc.read().decode("utf-8", "replace").split())[:300]
            raise WriterUnavailable(f"HTTP {exc.code}: {detail}") from None
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise WriterUnavailable(f"could not reach {url.split('/')[2]}: {exc}") from None
    raise AssertionError("unreachable")


class GeminiWriter:
    """Google Gemini through the public REST API."""

    URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

    def __init__(self, api_key: str | None = None, model: str | None = None, timeout: float = 60):
        self.api_key = api_key or env("GEMINI_API_KEY")
        self.model = model or env("GEMINI_MODEL") or "gemini-3.1-flash-lite"
        self.timeout = timeout
        self.name = f"gemini:{self.model}"

    def complete(self, system: str, prompt: str) -> Reply:
        if not self.api_key:
            raise WriterUnavailable("GEMINI_API_KEY is not set (see .env.example)")
        config: dict = {"temperature": 0.4, "maxOutputTokens": 1024}
        if "flash" in self.model:
            # Short text needs no thinking, and thinking tokens count against
            # the output budget.
            config["thinkingConfig"] = {"thinkingBudget": 0}
        data = _post(
            self.URL.format(model=self.model),
            {
                "systemInstruction": {"parts": [{"text": system}]},
                "contents": [{"role": "user", "parts": [{"text": prompt}]}],
                "generationConfig": config,
            },
            {"x-goog-api-key": self.api_key},
            self.timeout,
        )
        try:
            parts = data["candidates"][0]["content"]["parts"]
            text = "".join(p.get("text", "") for p in parts)
        except (KeyError, IndexError):
            raise WriterUnavailable(f"no text in the reply: {json.dumps(data)[:300]}") from None
        usage = data.get("usageMetadata", {})
        return Reply(text, usage.get("promptTokenCount"), usage.get("candidatesTokenCount"))


class OllamaWriter:
    """An open-source model running locally under Ollama."""

    def __init__(self, model: str | None = None, host: str | None = None, timeout: float = 180):
        self.model = model or env("OLLAMA_MODEL") or "llama3"
        self.host = (host or env("OLLAMA_HOST") or "http://localhost:11434").rstrip("/")
        self.timeout = timeout
        self.name = f"ollama:{self.model}"

    def complete(self, system: str, prompt: str) -> Reply:
        data = _post(
            f"{self.host}/api/generate",
            {
                "model": self.model,
                "system": system,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.4},
            },
            {},
            self.timeout,
        )
        return Reply(data.get("response", ""), data.get("prompt_eval_count"), data.get("eval_count"))


WRITERS = {"gemini": GeminiWriter, "ollama": OllamaWriter}


def get_writer(name: str) -> Writer | None:
    """A writer by name. "template" means no model: the fixed wording is used."""
    if name == "template":
        return None
    try:
        return WRITERS[name]()
    except KeyError:
        raise ValueError(f"unknown writer {name!r}; choose template, {', '.join(WRITERS)}") from None
