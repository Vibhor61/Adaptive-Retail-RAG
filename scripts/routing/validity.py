import enum
import math
import re
import unicodedata

from __future__ import annotations
from collections import Counter
from dataclasses import dataclass, field


MIN_QUERY_LENGTH = 3
MAX_QUERY_LENGTH = 500

MAX_SYMBOL_RATIO = 0.60
MAX_WHITESPACE_RATIO = 0.40


CONTROL_CHAR_PATTERN = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F]")
REPEATED_SYMBOL_PATTERN = re.compile(r"([^\w\s])\1{6,}")
REPEATED_CHAR_PATTERN = re.compile(r"(.)\1{10,}")


class StructuralStatus(enum.Enum):

    EXECUTABLE = "executable"

    SUSPICIOUS = "suspicious"

    DEGRADED = "degraded"


class StructuralFlag(enum.Enum):

    EMPTY_QUERY = "empty_query"

    EXCESSIVE_LENGTH = "excessive_length"

    CONTROL_CHARACTERS = "control_characters_detected"

    SYMBOL_SPAM = "symbol_spam_detected"

    CHARACTER_FLOOD = "character_flood_detected"

    TOKEN_REPETITION = "token_repetition_detected"

    HIGH_SYMBOL_RATIO = "high_symbol_ratio"

    HIGH_WHITESPACE_RATIO = "high_whitespace_ratio"

    LOW_STRUCTURAL_HEALTH = "low_structural_health"


@dataclass(frozen=True)
class StructuralSignals:

    entropy_score: float

    symbol_ratio: float

    alpha_ratio: float

    whitespace_ratio: float

    repetition_score: float

    length_score: float

    token_diversity_score: float 


@dataclass(frozen=True)
class StructuralValidationResult:

    status: StructuralStatus

    structural_health: float

    word_count: int

    normalized_query: str

    signals: StructuralSignals

    anomaly_flags: list[StructuralFlag] = field(default_factory=list)


def normalize_query(query: str) -> str:
    query = unicodedata.normalize("NFKC", query)
    query = query.strip()
    query = re.sub(r"\s+", " ", query)

    return query


def shannon_entropy(text: str) -> float:
    if not text:
        return 0.0

    counts = Counter(text)
    total = len(text)
    entropy = 0.0

    for count in counts.values():
        probability = count / total
        entropy -= probability * math.log2(probability)

    return entropy


def compute_symbol_ratio(text: str) -> float:
    if not text:
        return 0.0

    symbols = sum(
        1 for c in text if not c.isalnum() and not c.isspace()
    )

    return symbols / len(text)


def compute_alpha_ratio(text: str) -> float:
    if not text:
        return 0.0

    alpha = sum(1 for c in text if c.isalpha())

    return alpha / len(text)


def compute_whitespace_ratio(text: str) -> float:
    if not text:
        return 0.0

    whitespace = sum(1 for c in text if c.isspace())

    return whitespace / len(text)


def compute_repetition_score(text: str) -> float:
    """
    Higher means healthier diversity.
    Lower means excessive repetition.
    """

    if not text:
        return 0.0

    counts = Counter(text)
    most_common = counts.most_common(1)[0][1]

    repetition_ratio = most_common / len(text)

    return 1.0 - repetition_ratio



def validate_query_structure(query: str) -> StructuralValidationResult:

    normalized = normalize_query(query)

    anomaly_flags = []

    if not normalized:
        return StructuralValidationResult(
            status=StructuralStatus.DEGRADED,
            structural_health=0.0,
            reason="empty query",
            normalized_query="",
            signals=StructuralSignals(
                length_score=0.0,
                entropy_score=0.0,
                symbol_ratio=0.0,
                alpha_ratio=0.0,
                repetition_score=0.0,
                whitespace_ratio=0.0,
            ),
            anomaly_flags=["empty_input"]
        )

    if len(normalized) > MAX_QUERY_LENGTH:
        anomaly_flags.append("oversized_input")

    if CONTROL_CHAR_PATTERN.search(normalized):
        anomaly_flags.append("control_characters_detected")

    if REPEATED_SYMBOL_PATTERN.search(normalized):
        anomaly_flags.append("symbol_spam_detected")

    if REPEATED_CHAR_PATTERN.search(normalized):
        anomaly_flags.append("character_flood_detected")

    entropy = shannon_entropy(normalized)

    entropy_score = min(entropy / 5.0, 1.0)

    length_score = min(len(normalized) / 20.0, 1.0)

    symbol_ratio = compute_symbol_ratio(normalized)

    alpha_ratio = compute_alpha_ratio(normalized)

    whitespace_ratio = compute_whitespace_ratio(normalized)

    repetition_score = compute_repetition_score(normalized)

    signals = StructuralSignals(
        length_score=length_score,
        entropy_score=entropy_score,
        symbol_ratio=symbol_ratio,
        alpha_ratio=alpha_ratio,
        repetition_score=repetition_score,
        whitespace_ratio=whitespace_ratio,
    )

    quality_score = (
        0.30 * entropy_score +
        0.25 * repetition_score +
        0.20 * length_score +
        0.15 * (1.0 - min(symbol_ratio, 1.0)) +
        0.10 * (1.0 - whitespace_ratio)
    )

    # severe degradation
    if (
        "control_characters_detected" in anomaly_flags
        or quality_score < 0.20
    ):
        return StructuralValidationResult(
            status=StructuralStatus.DEGRADED,
            structural_health=max(0.05, quality_score),
            reason="structurally degraded input",
            normalized_query=normalized,
            signals=signals,
            anomaly_flags=anomaly_flags
        )

    # suspicious but executable
    if ( quality_score < 0.45 or len(anomaly_flags) > 0 ):
        return StructuralValidationResult(
            status=StructuralStatus.SUSPICIOUS,
            structural_health=quality_score,
            reason="suspicious structural patterns detected",
            normalized_query=normalized,
            signals=signals,
            anomaly_flags=anomaly_flags
        )

    return StructuralValidationResult(
        status=StructuralStatus.EXECUTABLE,
        structural_health=quality_score,
        reason="query structure acceptable",
        normalized_query=normalized,
        signals=signals,
        anomaly_flags=anomaly_flags
    )