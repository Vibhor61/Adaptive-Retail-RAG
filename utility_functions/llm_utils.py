"""
Utility module for the RAG pipeline.
Provides helper functions for processing and normalizing text.
Includes functions to safely parse and extract JSON data from language model outputs.
"""
import json
import re
from typing import Any, Dict, Optional, Set


def normalize_llm_text(text: str) -> str:
    """
    Strips whitespace and removes markdown formatting from language model output.
    Returns the cleaned text string.
    """
    if not text:
        return ""
    text = text.strip()
    text = re.sub(r"```json|```", "", text).strip()
    return text


def extract_json_from_text(text: str) -> Dict[str, Any]:
    """
    Attempts to extract and parse a JSON object from a given text string.
    Falls back to regex extraction and cleanup if standard JSON parsing fails.
    """

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
    """
    Extracts a specific key-value pair from a text string using regex.
    Returns the matched value or None if not found.
    """
    if not text:
        return None

    pattern = rf"{key}\s*=\s*([a-zA-Z0-9_]+)"
    match = re.search(pattern, text.strip())

    if match: 
        return match.group(1)
    else: 
        return None


def extract_citation_ids(text: str) -> Set[str]:
    """
    Extracts citation IDs (e.g. [CTX_1]) from a text string.
    Returns a set of extracted citation IDs.
    """
    if not text:
        return set()
    CTX_PATTERN = re.compile(r"\[CTX_(\d+)\]")
    return set(CTX_PATTERN.findall(text))


def parse_llm_output(text: str, mode: str = "json") -> Dict[str, Any]:
    """
    Parses language model output based on the specified extraction mode.
    Supports raw, key-value, and JSON modes.
    """
    if mode == "raw":
        return {"raw": text}

    if mode == "kv":
        return {"raw": text}

    if mode == "json":
        return extract_json_from_text(text)

    raise ValueError("invalid mode")


def safe_llm_call(llm, prompt: str, mode: str = "json", config: Optional[Dict] = None) -> Dict[str, Any]:
    """
    Executes a language model call safely and parses the output.
    Accepts an LLM instance, a prompt, and an extraction mode.
    """
    kwargs = {}
    if config:
        kwargs["config"] = config
    raw = llm.invoke(prompt, **kwargs)

    text = raw if isinstance(raw, str) else getattr(raw, "content", str(raw))

    return parse_llm_output(text, mode=mode)
