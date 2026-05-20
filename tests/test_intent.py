import pytest
from unittest.mock import patch, MagicMock
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))


from router.intent import (
    analyze_intent,
    llm_fallback,
    QueryType
)
import json
from collections import Counter


# def test_product_lookup_intent():
#     res = analyze_intent("iphone price cost")

#     assert res.query_type == QueryType.PRODUCT_LOOKUP

#     assert 0.0 <= res.confidence <= 1.0
#     assert isinstance(res.llm_used, bool)

# def test_comparison_intent():
#     res = analyze_intent("iphone vs samsung compare")

#     assert res.query_type == QueryType.COMPARISON
#     assert res.llm_used is False
#     assert res.confidence >= 0.7


# def test_review_intent():
#     res = analyze_intent("best laptop review worth buying")

#     assert res.query_type == QueryType.REVIEW_QUERY
#     assert res.llm_used is False


# import json
# from collections import Counter

# from router.intent import analyze_intent, QueryType


def run_eval(path="eval/intent_dataset.json"):
    with open(path, "r") as f:
        data = json.load(f)

    stats = Counter()
    llm_hits = 0
    correct = 0

    for item in data:
        query = item["query"]
        expected = QueryType[item["expected"]]

        result = analyze_intent(query)

        stats[result.query_type.value] += 1

        if result.llm_used:
            llm_hits += 1

        if result.query_type == expected:
            correct += 1

    total = len(data)

    print("\n=== INTENT EVALUATION REPORT ===")
    print(f"Total queries: {total}")
    print(f"Accuracy: {correct / total:.2f}")
    print(f"LLM fallback rate: {llm_hits / total:.2f}")

    print("\nIntent distribution:")
    for k, v in stats.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    run_eval()