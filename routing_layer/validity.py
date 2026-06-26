"""
Handles validation and sanitization of raw input queries.
Cleans up HTML and typographic artifacts, checks for anomalies like excessive length
or symbol spam, and evaluates the query's structural validity.
"""
import re
import html
import unicodedata

from contracts.router_contracts import(
    ValidationResult,
    ValidityStatus,
    ValidityFlags
)


MAX_QUERY_LENGTH = 500
HARD_REJECT_LENGTH = 2000

MAX_SYMBOL_RATIO = 0.45

CONTROL_CHAR_PATTERN = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F]")
REPEATED_SYMBOL_PATTERN = re.compile(r"([^\w\s])\1{3,}")
REPEATED_CHAR_PATTERN = re.compile(r"(.)\1{6,}")


HTML_TAG_PATTERN = re.compile(r"<[^>]+>")
 
TYPOGRAPHIC_FOLD_MAP = {
    "\u2018": "'",   # ‘ left single quote
    "\u2019": "'",   # ’ right single quote / apostrophe
    "\u201C": '"',   # “ left double quote
    "\u201D": '"',   # ” right double quote
    "\u2013": "-",   # – en dash
    "\u2014": "-",   # — em dash
    "\u00A0": " ",   # non-breaking space
    "\u200B": "",    # zero-width space
}

TYPOGRAPHIC_FOLD_PATTERN = re.compile("|".join(re.escape(k) for k in TYPOGRAPHIC_FOLD_MAP))


def clean_text_artifacts(text: str) -> str:
    """
    Removes HTML tags, unescapes HTML entities, and normalizes typographic artifacts.
    Replaces special quotes, dashes, and spaces with standard ASCII equivalents.
    """
    if not text:
        return text
 
    text = HTML_TAG_PATTERN.sub(" ", text)
    text = html.unescape(text)
    text = TYPOGRAPHIC_FOLD_PATTERN.sub(lambda m: TYPOGRAPHIC_FOLD_MAP[m.group(0)], text)
 
    return text


def _degraded(flag: ValidityFlags, normalized: str, word_count: int) -> ValidationResult:
    """
    Constructs a ValidationResult with a DEGRADED status and the specified anomaly flag.
    Used for early exits when a query fails basic structural checks.
    """
    return ValidationResult(
        status=ValidityStatus.DEGRADED,
        normalized_query=normalized,
        word_count=word_count,
        anomaly_flags=[flag],
    )

def normalize_query(query: str) -> str:
    """
    Applies text artifact cleaning, Unicode normalization, and whitespace collapse.
    Returns a clean, standardized string ready for downstream processing.
    """

    query = clean_text_artifacts(query)
    query = unicodedata.normalize("NFKC", query)
    query = query.strip()
    query = re.sub(r"\s+", " ", query)

    return query


def compute_symbol_ratio(text: str) -> float:
    """
    Calculates the proportion of non-alphanumeric, non-whitespace characters in a string.
    Returns a float representing the symbol ratio.
    """
    if not text:
        return 0.0

    symbols = sum(1 for c in text if not c.isalnum() and not c.isspace())

    return symbols / len(text)


def validate_query_structure(query: str) -> ValidationResult:
    """
    Evaluates the structural integrity of a query based on length, symbols, and artifacts.
    Returns a ValidationResult with a classification of EXECUTABLE, SUSPICIOUS, or DEGRADED.
    """

    normalized = normalize_query(query)
    
    if not normalized:
        return _degraded(ValidityFlags.EMPTY_QUERY, "", 0)

    if len(normalized) > HARD_REJECT_LENGTH:
        return _degraded(ValidityFlags.HARD_LENGTH_REJECT, normalized[:100], len(normalized.split()))

    if CONTROL_CHAR_PATTERN.search(normalized):
        return _degraded(ValidityFlags.CONTROL_CHARACTERS, normalized, 0)
 
    word_count = len(normalized.split())
    anomaly_flags: list[ValidityFlags] = []

    if len(normalized) > MAX_QUERY_LENGTH:
        anomaly_flags.append(ValidityFlags.EXCESSIVE_LENGTH)

    if REPEATED_SYMBOL_PATTERN.search(normalized):
        anomaly_flags.append(ValidityFlags.SYMBOL_SPAM)

    if REPEATED_CHAR_PATTERN.search(normalized):
        anomaly_flags.append(ValidityFlags.CHARACTER_FLOOD)

    symbol_ratio = compute_symbol_ratio(normalized)

    if symbol_ratio > MAX_SYMBOL_RATIO:
        anomaly_flags.append(ValidityFlags.HIGH_SYMBOL_RATIO)
  
    if anomaly_flags:
        return ValidationResult(
            status=ValidityStatus.SUSPICIOUS,
            normalized_query=normalized,
            word_count=word_count,
            anomaly_flags=anomaly_flags
        )

    return ValidationResult(
        status=ValidityStatus.EXECUTABLE,
        normalized_query=normalized,
        word_count=word_count,
        anomaly_flags=[]
    )