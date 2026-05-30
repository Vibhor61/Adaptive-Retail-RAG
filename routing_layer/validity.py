import enum
import re
import unicodedata

from dataclasses import dataclass, field


MIN_QUERY_LENGTH = 1
MAX_QUERY_LENGTH = 500

MAX_SYMBOL_RATIO = 0.70
MAX_WHITESPACE_RATIO = 0.50

CONTROL_CHAR_PATTERN = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F]")
REPEATED_SYMBOL_PATTERN = re.compile(r"([^\w\s])\1{6,}")
REPEATED_CHAR_PATTERN = re.compile(r"(.)\1{10,}")


class ValidityStatus(enum.Enum):
    EXECUTABLE = "executable"

    SUSPICIOUS = "suspicious"

    DEGRADED = "degraded"


class ValidityFlags(enum.Enum):
    EMPTY_QUERY = "empty_query"

    EXCESSIVE_LENGTH = "excessive_length"

    CONTROL_CHARACTERS = "control_characters_detected"

    SYMBOL_SPAM = "symbol_spam_detected"

    CHARACTER_FLOOD = "character_flood_detected"

    HIGH_SYMBOL_RATIO = "high_symbol_ratio"

    HIGH_WHITESPACE_RATIO = "high_whitespace_ratio"


@dataclass(frozen=True)
class ValidationResult:
    status: ValidityStatus

    reason: str

    normalized_query: str

    word_count: int

    anomaly_flags: list[ValidityFlags] = field(default_factory=list)


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


def compute_whitespace_ratio(text: str) -> float:
    if not text:
        return 0.0

    whitespace = sum(1 for c in text if c.isspace())

    return whitespace / len(text)


def validate_query_structure(query: str) -> ValidationResult:

    normalized = normalize_query(query)
    anomaly_flags: list[ValidityFlags] = []

    if not normalized:
        return ValidationResult(
            status=ValidityStatus.DEGRADED,
            reason="empty query",
            normalized_query="",
            word_count=0,
            anomaly_flags=[
                ValidityFlags.EMPTY_QUERY
            ]
        )

    if len(normalized) > MAX_QUERY_LENGTH:
        anomaly_flags.append(ValidityFlags.EXCESSIVE_LENGTH)

    if CONTROL_CHAR_PATTERN.search(normalized):
        anomaly_flags.append(ValidityFlags.CONTROL_CHARACTERS)

    if REPEATED_SYMBOL_PATTERN.search(normalized):
        anomaly_flags.append(ValidityFlags.SYMBOL_SPAM)

    if REPEATED_CHAR_PATTERN.search(normalized):
        anomaly_flags.append(ValidityFlags.CHARACTER_FLOOD)

    symbol_ratio = compute_symbol_ratio(normalized)

    if symbol_ratio > MAX_SYMBOL_RATIO:
        anomaly_flags.append(ValidityFlags.HIGH_SYMBOL_RATIO)

    whitespace_ratio = compute_whitespace_ratio(normalized)

    if whitespace_ratio > MAX_WHITESPACE_RATIO:
        anomaly_flags.append(ValidityFlags.HIGH_WHITESPACE_RATIO)

    word_count = len(normalized.split())

    degraded_flags = {ValidityFlags.CONTROL_CHARACTERS}

    if any(flag in degraded_flags for flag in anomaly_flags):
        return ValidationResult(
            status=ValidityStatus.DEGRADED,
            reason="structurally degraded input",
            normalized_query=normalized,
            word_count=word_count,
            anomaly_flags=anomaly_flags
        )

    if anomaly_flags:
        return ValidationResult(
            status=ValidityStatus.SUSPICIOUS,
            reason="suspicious structural patterns detected",
            normalized_query=normalized,
            word_count=word_count,
            anomaly_flags=anomaly_flags
        )

    return ValidationResult(
        status=ValidityStatus.EXECUTABLE,
        reason="query structure acceptable",
        normalized_query=normalized,
        word_count=word_count,
        anomaly_flags=[]
    )