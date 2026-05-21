"""Interactive helper for the Ollama local-LLM backend.

Walks through:
  1. Verifying the `ollama` binary is installed
  2. Checking that the Ollama server is reachable on localhost:11434
  3. Listing currently-pulled models
  4. Offering to pull the default model (qwen3.5:9b — vision + tools, ~6.6GB)
  5. Running a small smoke-test prompt to confirm end-to-end success
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from urllib.error import URLError
from urllib.request import Request, urlopen


DEFAULT_BASE_URL = "http://localhost:11434"
DEFAULT_MODEL = "qwen3.5:9b"
PULL_SIZE_NOTE = "≈6.6GB download; vision + tool calling"
FALLBACK_MODELS = ("qwen3-vl:8b", "qwen3.5:4b")


def _check_binary() -> bool:
    if shutil.which("ollama"):
        print("✓ `ollama` binary on PATH")
        return True
    print("✗ `ollama` binary not found.")
    print("  Install: `brew install ollama`  (or download from https://ollama.com/download)")
    return False


def _check_server(base_url: str) -> bool:
    try:
        with urlopen(f"{base_url}/api/tags", timeout=2) as resp:
            if resp.status == 200:
                print(f"✓ Ollama server reachable at {base_url}")
                return True
    except (URLError, TimeoutError, OSError):
        pass
    print(f"✗ Ollama server not reachable at {base_url}.")
    print("  In another terminal, run:  `ollama serve`")
    return False


def _installed_models(base_url: str) -> list[str]:
    with urlopen(f"{base_url}/api/tags", timeout=5) as resp:
        body = json.load(resp)
    return [m["name"] for m in body.get("models", [])]


def _pull_model(model: str) -> bool:
    print(f"\nPulling {model} ({PULL_SIZE_NOTE}) — this can take a while…")
    proc = subprocess.run(["ollama", "pull", model])
    return proc.returncode == 0


def _smoke_test(base_url: str, model: str) -> bool:
    print(f"\nSmoke-testing {model}…")
    payload = json.dumps(
        {
            "model": model,
            "messages": [{"role": "user", "content": "Say 'ready' in one word."}],
            "stream": False,
        }
    ).encode("utf-8")
    req = Request(
        f"{base_url}/v1/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json", "Authorization": "Bearer ollama"},
    )
    try:
        with urlopen(req, timeout=120) as resp:
            body = json.load(resp)
        content = body["choices"][0]["message"]["content"]
        print(f"  → {content.strip()!r}")
        return True
    except (URLError, KeyError, json.JSONDecodeError) as e:
        print(f"  smoke test failed: {e}")
        return False


def _prompt_yes_no(question: str, default: bool = True) -> bool:
    suffix = " [Y/n] " if default else " [y/N] "
    try:
        ans = input(question + suffix).strip().lower()
    except EOFError:
        return default
    if not ans:
        return default
    return ans in {"y", "yes"}


def main() -> int:
    if not _check_binary():
        return 1
    if not _check_server(DEFAULT_BASE_URL):
        return 1

    models = _installed_models(DEFAULT_BASE_URL)
    print(f"\nInstalled models: {models if models else '(none)'}")

    if DEFAULT_MODEL not in models:
        print(f"Default model: {DEFAULT_MODEL} ({PULL_SIZE_NOTE})")
        print(f"Smaller fallbacks if it's too slow: {', '.join(FALLBACK_MODELS)}")
        if not _prompt_yes_no(f"Pull {DEFAULT_MODEL}?"):
            print("Skipped pull. Set llm.ollama.model in configs/default.yaml to one you have.")
            return 0
        if not _pull_model(DEFAULT_MODEL):
            print("Pull failed.")
            return 1
    else:
        print(f"✓ {DEFAULT_MODEL} already installed")

    if not _smoke_test(DEFAULT_BASE_URL, DEFAULT_MODEL):
        return 1

    print("\nAll set. Switch backends by editing configs/default.yaml:")
    print('  llm.provider: "ollama"')
    print(f'  llm.model: "{DEFAULT_MODEL}"')
    return 0


if __name__ == "__main__":
    sys.exit(main())
