import re
import math
import enum
from collections import Counter
from dataclasses import dataclass


class QueryValidity(enum.Enum):
    VALID = "valid"
    INVALID = "invalid"


@dataclass
class ValidationResult:
    validity: QueryValidity
    confidence: float
    reason: str


MIN_QUERY_LENGTH = 3

VOWELS = set('aeiou')


def vowel_ratio(query:str):
    letters = [c for c in query.lower() if c.isalpha()]

    if not letters:
        return 0.0
    
    vowel_count = 0
    
    for l in letters:
        if l in VOWELS:
            vowel_count += 1
        
    return vowel_count/len(letters)


def character_entropy(text: str) -> float:
    counts = Counter(text)
    total = len(text)
    entropy = 0.0

    for count in counts.values():
        probability = count / total
        entropy -= probability * math.log2(probability)
    
    return entropy


def is_query_valid(query: str) -> ValidationResult:

    cleaned = query.strip().lower()

    if not cleaned:
        return ValidationResult(
            validity=QueryValidity.INVALID,
            confidence=0.99,
            reason="empty query"
        )

    if len(cleaned) < MIN_QUERY_LENGTH:
        return ValidationResult(
            validity=QueryValidity.INVALID,
            confidence=0.90,
            reason="query too short"
        )
    
    if re.fullmatch(r"[\W\d_]+", cleaned):
        return ValidationResult(
            validity=QueryValidity.INVALID,
            confidence=0.98,
            reason="only symbols or numbers"
        )

    if len(set(cleaned)) == 1:
        return ValidationResult(
            validity=QueryValidity.INVALID,
            confidence=0.97,
            reason="single repeated character"
        )
    

    alpha_text = "".join(c for c in cleaned if c.isalpha())

    # gibberish heuristic
    if alpha_text and len(alpha_text) >= 6:

        vr = vowel_ratio(alpha_text)

        if vr < 0.15:
            return ValidationResult(
                validity=QueryValidity.INVALID,
                confidence=0.80,
                reason="low vowel ratio probable gibberish"
            )

    # entropy heuristic
    entropy = character_entropy(cleaned)

    if entropy < 1.2:
        return ValidationResult(
            validity=QueryValidity.INVALID,
            confidence=0.75,
            reason="low entropy repetitive query"
        )

    return ValidationResult(
        validity=QueryValidity.VALID,
        confidence=0.92,
        reason="query passed validation"
    )

