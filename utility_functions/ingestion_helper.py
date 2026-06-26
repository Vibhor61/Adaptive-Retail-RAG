"""
Helper functions for the data ingestion pipeline.

This module provides utility functions for reading compressed JSONL files,
normalizing text and prices, generating stable hashes, and extracting
structured metadata and reviews from raw JSON objects.
"""

import json
import gzip
from typing import  Optional, Tuple
import hashlib
import re


def extract_shard(path:str) :
    """
    Opens a file, automatically handling gzip decompression if needed.
    Returns a text file object.
    """
    if path.endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8", errors="ignore")
    return open(path, "rt", encoding="utf-8", errors="ignore")


def iter_rows(path : str) :
    """
    Iterates over lines in a JSONL file and yields parsed JSON objects.
    Skips empty lines and lines with invalid JSON.
    """
    with extract_shard(path) as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue 


def stable_hash(*parts: str) -> str:
    """
    Generates a stable SHA1 hash from multiple string parts.
    Useful for creating deterministic IDs like review IDs.
    """
    h = hashlib.sha1()
    for p in parts:
        h.update((p or "").encode("utf-8", errors="ignore"))
        h.update(b"\x1f")  # separator
    return h.hexdigest()


WS = re.compile(r"\s+")
def norm_text(x) -> str:
    """
    Normalizes text by removing extra whitespace and converting to string.
    """
    if x is None:
        return ""
    s = str(x).strip()
    s = WS.sub(" ", s)
    return s

PRICE_CLEAN = re.compile(r"[^\d.]")
def norm_price(x) -> Optional[float]:
    """
    Parses and normalizes price strings into float values.
    Handles ranges and strips non-numeric characters.
    """
    if x is None:
        return None

    if isinstance(x, list):
        if not x:
            return None
        x = x[0]

    s = norm_text(x)

    if not s:
        return None
    
    if "-" in s:
        s = s.split("-")[0]

    s = PRICE_CLEAN.sub("", s)
    if not s:
        return None

    try:
        return float(s)
    except ValueError:
        return None


def normalize_list_field(value) -> list[str]:
    """
    Normalizes a list or string field into a list of cleaned strings.
    """
    if value is None:
        return []
    
    if isinstance(value, list):
        return [norm_text(item) for item in value if norm_text(item)]
    
    if isinstance(value, str):
        cleaned = norm_text(value)
        return [cleaned] if cleaned else []

    return []


def normalize_brand(value) -> str:
    """
    Normalizes a brand string, mapping empty or generic missing values to empty string.
    """
    brand = norm_text(value)

    if brand.lower() in {
        "", "n/a", "none", "null", "unknown"
    }:
        return ""

    return brand

def write_jsonl(path: str, rows) -> None:
    """
    Writes a list of dictionaries to a JSON lines file.
    """
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def extract_metadata(obj: dict) -> Optional[Tuple[str, dict]]:
    """
    Extracts and normalizes product metadata from a raw JSON object.
    Returns the ASIN and a dictionary of cleaned fields, or None if invalid.
    """
    asin = norm_text(obj.get("asin"))
    if not asin:
        return None
    
    category = normalize_list_field(
        obj.get("category") or obj.get("categories")
    )

    description = normalize_list_field(
        obj.get("description")
    )

    feature = normalize_list_field(
        obj.get("feature")
    )

    keep = {
        "asin": asin,
        "title": norm_text(obj.get("title")),
        "brand": normalize_brand(obj.get("brand")),
        "category": category,
        "feature": feature,
        "main_cat": norm_text(obj.get("main_cat")),
        "description": description,
        "price": norm_price(obj.get("price")),
        "price_raw" : obj.get("price")
    }

    return asin, keep


def extract_reviews(obj : dict):
    """
    Extracts and normalizes review data from a raw JSON object.
    Generates a stable review_id based on content and returns a dictionary.
    """

    asin = norm_text(obj.get("asin"))
    if not asin:
        return None

    review_text = norm_text(obj.get("reviewText") or obj.get("review_text"))
    summary_text = norm_text(obj.get("summary") or obj.get("summary_text"))
    ts = norm_text(obj.get("unixReviewTime") or obj.get("unix_review_time"))

    if not review_text and not summary_text:
        return None
    
    review_id = stable_hash(asin,review_text,summary_text,ts)

    return {
        "review_id":review_id,
        "asin" : asin,
        "review_text" : review_text,
        "summary_text" : summary_text
    }
    