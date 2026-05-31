import enum
import re
import unicodedata

from dataclasses import dataclass, field


MAX_QUERY_LENGTH = 500
HARD_REJECT_LENGTH = 2000

MAX_SYMBOL_RATIO = 0.45

CONTROL_CHAR_PATTERN = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F]")
REPEATED_SYMBOL_PATTERN = re.compile(r"([^\w\s])\1{3,}")
REPEATED_CHAR_PATTERN = re.compile(r"(.)\1{6,}")


class ValidityStatus(enum.Enum):
    EXECUTABLE = "executable"

    SUSPICIOUS = "suspicious"

    DEGRADED = "degraded"


class ValidityFlags(enum.Enum):
    EMPTY_QUERY = "empty_query"

    EXCESSIVE_LENGTH = "excessive_length"

    HARD_LENGTH_REJECT = "hard_length_reject"

    CONTROL_CHARACTERS = "control_characters_detected"

    SYMBOL_SPAM = "symbol_spam_detected"

    CHARACTER_FLOOD = "character_flood_detected"

    HIGH_SYMBOL_RATIO = "high_symbol_ratio"


@dataclass(frozen=True)
class ValidationResult:
    status: ValidityStatus

    normalized_query: str

    word_count: int

    anomaly_flags: list[ValidityFlags] = field(default_factory=list)


def _degraded(flag: ValidityFlags, normalized: str, word_count: int) -> ValidationResult:
    return ValidationResult(
        status=ValidityStatus.DEGRADED,
        normalized_query=normalized,
        word_count=word_count,
        anomaly_flags=[flag],
    )

def normalize_query(query: str) -> str:

    query = unicodedata.normalize("NFKC", query)
    query = query.strip()
    query = re.sub(r"\s+", " ", query)

    return query


def compute_symbol_ratio(text: str) -> float:
    if not text:
        return 0.0

    symbols = sum(1 for c in text if not c.isalnum() and not c.isspace())

    return symbols / len(text)


def validate_query_structure(query: str) -> ValidationResult:

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