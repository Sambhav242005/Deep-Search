import json
import random
import sys
import time
from typing import Any, Dict
import requests

from backend.constant import CHAT_ENDPOINT, GEN_ENDPOINT, REQUEST_TIMEOUT


def _ask_ollama(
    model: str,
    prompt: str,
    *,
    system: str | None = None,
    fmt: dict | str | None = None,
    temperature: float = 0.1,
    max_retries: int = 3,
) -> str:
    """Call Ollama with retry logic and enhanced error handling."""
    is_schema = isinstance(fmt, dict)
    url = CHAT_ENDPOINT if is_schema else GEN_ENDPOINT

    if is_schema:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        body: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": False,
            "format": fmt,
            "options": {"temperature": temperature}
        }
    else:
        body = {
            "model": model, 
            "prompt": prompt, 
            "stream": False,
            "options": {"temperature": temperature}
        }
        if system:
            body["system"] = system
        if fmt is not None:
            body["format"] = fmt

    last_error = None
    for attempt in range(max_retries):
        try:
            r = requests.post(url, json=body, timeout=REQUEST_TIMEOUT)
            
            if r.status_code >= 400:
                if r.status_code == 404:
                    raise RuntimeError(f"Model '{model}' not found. Available models can be listed with 'ollama list'")
                raise RuntimeError(f"Ollama HTTP {r.status_code}: {r.text[:300]}")

            try:
                data = r.json()
            except ValueError:
                raise RuntimeError(f"Non-JSON response from Ollama: {r.text[:300]}")

            if "error" in data:
                raise RuntimeError(f"Ollama error: {data['error']}")

            if is_schema:
                content = data.get("message", {}).get("content", "")
                return json.dumps(content) if isinstance(content, (dict, list)) else str(content)

            if "response" in data:
                return str(data["response"]).strip()

            choices = data.get("choices")
            if choices and isinstance(choices, list):
                return str(choices[0].get("text", "")).strip()

            raise RuntimeError("Unexpected Ollama JSON response structure")
            
        except requests.exceptions.RequestException as exc:
            last_error = RuntimeError(f"Cannot reach Ollama at {url}: {exc}")
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt + random.uniform(0, 1)
                print(f"[retry] Ollama request failed, retrying in {wait_time:.1f}s...", file=sys.stderr)
                time.sleep(wait_time)
            continue
        except Exception as exc:
            last_error = exc
            if attempt < max_retries - 1:
                time.sleep(1)
            continue
    
    raise last_error or RuntimeError("Unknown error in Ollama request")
