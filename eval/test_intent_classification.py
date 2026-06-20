import json
import os
import logging
import sys
import re
import statistics
import time

from pathlib import Path
from langchain_ollama import ChatOllama

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from routing_layer.intent_classification import IntentClassifier, Intent, EntityStructure

logging.getLogger("opentelemetry").setLevel(logging.ERROR)

llm = ChatOllama(
    model="phi3:latest",
    temperature=0,
    base_url="http://127.0.0.1:5105",
)

classifier = IntentClassifier(llm=llm)

dataset_path = os.path.join(CURRENT_DIR, "data", "intent_dataset.json")

with open(dataset_path, "r") as f:
    dataset = json.load(f)

def normalize(text: str):
    text = text.lower()
    text = re.sub(r"[^a-z0-9 ]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def tokens(text: str):
    return set(normalize(text).split())

def containment_score(gold, pred):
    g = normalize(gold)
    p = normalize(pred)
    return 1.0 if g in p or p in g else 0.0

def overlap_score(gold, pred):
    g = tokens(gold)
    p = tokens(pred)
    if not g:
        return 1.0 if not p else 0.0
    return len(g & p) / len(g)

latencies = []
intent_correct = 0
struct_correct = 0
containment_scores = []
overlap_scores = []

print("-" * 90)
print(f"Index | Query Preview | Intent (P/E) | Struct (P/E) | Cont | Over | Latency(ms)")
print("-" * 90)

for idx, item in enumerate(dataset, start=1):
    t0 = time.perf_counter()

    intent, entities, entity_structure = classifier.classify_intent(item["query"])

    latency = (time.perf_counter() - t0) * 1000
    latencies.append(latency)

    intent_ok = intent.value == item["expected_intent"]
    intent_correct += int(intent_ok)

    struct_ok = str(entity_structure.value).upper() == str(item["expected_structure"]).upper()
    struct_correct += int(struct_ok)

    gold_entities = item["expected_entities"]
    pred_entities = entities

    if not gold_entities and not pred_entities:
        final_cont = 1.0
        final_ov = 1.0
    elif not gold_entities and pred_entities:
        final_cont = 0.0
        final_ov = 0.0
    else:
        cont_scores = []
        ov_scores = []
        for g in gold_entities:
            best_cont = max([containment_score(g, p) for p in pred_entities] or [0.0])
            best_ov = max([overlap_score(g, p) for p in pred_entities] or [0.0])
            cont_scores.append(best_cont)
            ov_scores.append(best_ov)

        final_cont = sum(cont_scores) / len(cont_scores) if cont_scores else 1.0
        final_ov = sum(ov_scores) / len(ov_scores) if ov_scores else 1.0

    containment_scores.append(final_cont)
    overlap_scores.append(final_ov)

    query_preview = item["query"][:30] + "..." if len(item["query"]) > 30 else item["query"]
    intent_str = f"{intent.value}/{item['expected_intent']}"
    struct_str = f"{entity_structure.value}/{item['expected_structure']}"
    
    print(f"#{idx} | {query_preview} | {intent_str} | {struct_str} | {final_cont:.2f} | {final_ov:.2f} | {latency:.1f}")
    print(f"  Raw Query: {item['query']}")
    print(f"  Expected: {gold_entities}")
    print(f"  Extracted: {pred_entities}")
    if not struct_ok:
        print(f"  Mismatch: Got '{entity_structure.value}', Expected '{item['expected_structure']}'")
    print("-" * 50)

print("\n" + "=" * 50)
print(f"Total Test Cases: {len(dataset)}")
print(f"Intent Accuracy: {intent_correct / len(dataset):.3f}")
print(f"Entity Structure Accuracy: {struct_correct / len(dataset):.3f}")
print(f"Entity Containment Score: {sum(containment_scores)/len(containment_scores):.3f}")
print(f"Entity Overlap Score: {sum(overlap_scores)/len(overlap_scores):.3f}")
print(f"Latency: mean={statistics.mean(latencies):.1f}ms | median={statistics.median(latencies):.1f}ms")
print("=" * 50)