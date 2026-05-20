import json
import re
from typing import Any, Dict, Optional


def normalize_llm_text(text: str) -> str:
    if not text:
        return ""
    text = text.strip()
    text = re.sub(r"```json|```", "", text).strip()
    return text


def extract_json_from_text(text: str) -> Dict[str, Any]:

    if not text:
        raise ValueError("Empty LLM output")

    text = normalize_llm_text(text)

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except Exception:
            pass

    cleaned = re.sub(r",\s*}", "}", text)
    cleaned = re.sub(r",\s*]", "]", cleaned)

    return json.loads(cleaned)


def extract_key_value(text: str, key: str) -> Optional[str]:
    if not text:
        return None

    pattern = rf"{key}\s*=\s*([a-zA-Z0-9_]+)"
    match = re.search(pattern, text.strip())

    if match: 
        return match.group(1)
    else: 
        return None


def parse_llm_output(text: str, mode: str = "json") -> Dict[str, Any]:
    if mode == "raw":
        return {"raw": text}

    if mode == "kv":
        return {"raw": text}

    if mode == "json":
        return extract_json_from_text(text)

    raise ValueError("invalid mode")


def safe_llm_call(llm, prompt: str, mode: str = "json") -> Dict[str, Any]:
    raw = llm(prompt)

    text = raw if isinstance(raw, str) else getattr(raw, "content", str(raw))

    return parse_llm_output(text, mode=mode)