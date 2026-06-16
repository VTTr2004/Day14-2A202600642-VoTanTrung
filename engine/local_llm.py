import json
import os
import threading
import time
import urllib.error
import urllib.request
from typing import Dict, List, Optional

from dotenv import load_dotenv


load_dotenv()

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
DEFAULT_MODEL = os.getenv("AGENT_MODEL", "kamekichi128/qwen3-4b-instruct-2507:latest")
GEMINI_API_URL = os.getenv("GEMINI_API_URL", "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent")
LLM_CALL_DELAY_SECONDS = float(os.getenv("LLM_CALL_DELAY_SECONDS", "2"))
_LLM_DELAY_LOCK = threading.Lock()
_LAST_LLM_CALL_AT = 0.0
_GEMINI_LOCK = threading.Lock()
_GEMINI_KEY_INDEX = 0
_EXHAUSTED_GEMINI_KEYS = set()


class LocalLLMError(RuntimeError):
    pass


def _wait_before_llm_call() -> None:
    global _LAST_LLM_CALL_AT
    if LLM_CALL_DELAY_SECONDS <= 0:
        return
    with _LLM_DELAY_LOCK:
        now = time.monotonic()
        wait_for = LLM_CALL_DELAY_SECONDS - (now - _LAST_LLM_CALL_AT)
        if wait_for > 0:
            time.sleep(wait_for)
        _LAST_LLM_CALL_AT = time.monotonic()


def _mask_key(key: str) -> str:
    if len(key) <= 8:
        return "*" * len(key)
    return f"{key[:4]}...{key[-4:]}"


def _gemini_keys() -> List[str]:
    raw = os.getenv("GEMINI_API_KEY", "")
    keys = [key.strip() for key in raw.split(",") if key.strip()]
    if not keys:
        raise LocalLLMError("Thieu GEMINI_API_KEY trong file .env de goi Gemini judge.")
    return keys


def _is_retryable_key_error(status: int, body: str) -> bool:
    body_lower = body.lower()
    key_markers = [
        "api_key_invalid",
        "api key not valid",
        "api key expired",
        "api key disabled",
        "key not valid",
        "resource_exhausted",
        "quota",
        "rate limit",
        "rate_limit",
        "exceeded",
        "too many requests",
    ]
    return status == 429 or status in {400, 403} and any(marker in body_lower for marker in key_markers)


def _next_gemini_key(keys: List[str]) -> str:
    global _GEMINI_KEY_INDEX
    with _GEMINI_LOCK:
        for offset in range(len(keys)):
            index = (_GEMINI_KEY_INDEX + offset) % len(keys)
            key = keys[index]
            if key not in _EXHAUSTED_GEMINI_KEYS:
                _GEMINI_KEY_INDEX = index
                return key
    raise LocalLLMError("Tat ca Gemini API key trong .env da het quota hoac khong dung duoc.")


def _mark_gemini_key_exhausted(key: str, keys: List[str]) -> None:
    global _GEMINI_KEY_INDEX
    with _GEMINI_LOCK:
        _EXHAUSTED_GEMINI_KEYS.add(key)
        if keys:
            current = keys.index(key) if key in keys else _GEMINI_KEY_INDEX
            _GEMINI_KEY_INDEX = (current + 1) % len(keys)


def generate_local(
    prompt: str,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.0,
    num_predict: int = 256,
    timeout: int = 120,
) -> Dict:
    _wait_before_llm_call()
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": num_predict,
        },
    }
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        OLLAMA_URL,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            result = json.loads(response.read().decode("utf-8"))
    except Exception as exc:
        raise LocalLLMError(f"Khong goi duoc Ollama API tai {OLLAMA_URL}: {exc}") from exc

    return {
        "text": result.get("response", "").strip(),
        "model": result.get("model", model),
        "prompt_eval_count": result.get("prompt_eval_count", 0),
        "eval_count": result.get("eval_count", 0),
        "total_duration": result.get("total_duration", 0),
    }


def generate_gemini(
    prompt: str,
    model: str,
    temperature: float = 0.0,
    num_predict: int = 256,
    timeout: int = 120,
) -> Dict:
    _wait_before_llm_call()
    keys = _gemini_keys()
    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [{"text": prompt}],
            }
        ],
        "generationConfig": {
            "temperature": temperature,
            "maxOutputTokens": num_predict,
        },
    }
    data = json.dumps(payload).encode("utf-8")

    last_error = None
    for _ in range(len(keys)):
        api_key = _next_gemini_key(keys)
        url = GEMINI_API_URL.format(model=model)
        separator = "&" if "?" in url else "?"
        url = f"{url}{separator}key={api_key}"
        request = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                result = json.loads(response.read().decode("utf-8"))
            break
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            last_error = f"HTTP {exc.code} voi key {_mask_key(api_key)}: {body[:300]}"
            if _is_retryable_key_error(exc.code, body):
                _mark_gemini_key_exhausted(api_key, keys)
                continue
            raise LocalLLMError(f"Khong goi duoc Gemini API voi model {model}: {last_error}") from exc
        except Exception as exc:
            last_error = f"{type(exc).__name__} voi key {_mask_key(api_key)}: {exc}"
            raise LocalLLMError(f"Khong goi duoc Gemini API voi model {model}: {last_error}") from exc
    else:
        raise LocalLLMError(f"Tat ca Gemini API key da het quota, sai, hoac khong dung duoc. Loi cuoi: {last_error}")

    candidates = result.get("candidates", [])
    parts = candidates[0].get("content", {}).get("parts", []) if candidates else []
    text = "".join(part.get("text", "") for part in parts).strip()
    usage = result.get("usageMetadata", {})
    return {
        "text": text,
        "model": model,
        "prompt_eval_count": usage.get("promptTokenCount", 0),
        "eval_count": usage.get("candidatesTokenCount", 0),
        "total_duration": 0,
    }


def generate_model(
    prompt: str,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.0,
    num_predict: int = 256,
    timeout: int = 120,
) -> Dict:
    if model.lower().startswith("gemini"):
        return generate_gemini(prompt, model, temperature, num_predict, timeout)
    return generate_local(prompt, model, temperature, num_predict, timeout)


def extract_json_object(text: str) -> Optional[Dict]:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        return json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None
