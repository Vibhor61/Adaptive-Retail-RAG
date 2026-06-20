

import sys
import time
import statistics
import json
import os

from pathlib import Path
from langchain_ollama import ChatOllama

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from routing_layer.evidence_type_classification import EvidenceClassifier

# Make Sure loca ollama or docker ollama is running and model is pulled 
llm = ChatOllama(
    model="phi3:latest",
    temperature=0,
    base_url="http://127.0.0.1:5105", # Point to correct port
)

classifier = EvidenceClassifier(llm=llm)

results = []
latencies = []

print("=" * 75)
print(f"{'ID':<12} {'EXPECTED':<16} {'PREDICTED':<16} {'LATENCY':>8}  PASS")
print("=" * 75)

dataset_path = os.path.join(CURRENT_DIR, "data", "evidence_dataset.json")
with open(dataset_path, "r") as f:
    dataset = json.load(f)

for item in dataset:
    t0 = time.perf_counter()
    predicted_evidence = classifier.llm_evidence_classification(item["query"])
    elapsed_ms = (time.perf_counter() - t0) * 1000

    predicted = predicted_evidence.value.upper()
    expected = item["expected_evidence"]
    passed = predicted == expected

    latencies.append(elapsed_ms)
    results.append({
        "id": item["id"],
        "query": item["query"],
        "expected": expected,
        "predicted": predicted,
        "latency_ms": elapsed_ms,
        "passed": passed,
    })

    status = "PASS" if passed else "FAIL"
    print(
        f"{item['id']:<12} "
        f"{expected:<16} "
        f"{predicted:<16} "
        f"{elapsed_ms:>7.1f}ms  {status}"
    )

total = len(results)
correct = sum(r["passed"] for r in results)
failed = [r for r in results if not r["passed"]]

print("=" * 75)
print(f"Accuracy: {correct}/{total}  ({correct/total*100:.1f}%)")
print(
    f"Latency: "
    f"mean={statistics.mean(latencies):.1f}ms  "
    f"median={statistics.median(latencies):.1f}ms  "
    f"p95={sorted(latencies)[int(len(latencies)*0.95)]:.1f}ms  "
    f"n={len(latencies)}"
)

print()
for label in ("FACTUAL", "EXPERIENTIAL", "MIXED"):
    subset = [r for r in results if r["expected"] == label]
    correct_label = sum(r["passed"] for r in subset)
    print(f"{label:<16}: {correct_label}/{len(subset)}")

if failed:
    print(f"\nFailures ({len(failed)})")
    for r in failed:
        print(f"  [{r['id']}] expected={r['expected']}  got={r['predicted']}")
        print(f"  query: {r['query']}")

print("=" * 75)