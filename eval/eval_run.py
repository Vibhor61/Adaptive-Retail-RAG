"""
Simplified runner script for the RAG system's evaluation endpoint.
Loops through the dataset sequentially, hits the eval endpoint, and saves the raw outputs.
"""

import json
import time
import sys
from pathlib import Path
import requests
from config.settings import settings

SCRIPT_DIR = Path(__file__).resolve().parent

OUT_PATH = SCRIPT_DIR / "data" / "raw_outputs.json"
DATASET_PATH = SCRIPT_DIR / "data" / "eval_dataset.json"

BASE_URL = f"{settings.backend_url}/eval_chat"
LIMIT = 140
DELAY_BETWEEN_QUERIES_SECONDS = 2.0
TIMEOUT_SECONDS = 120

def load_dataset(path: Path) -> list[dict]:
    with open(path, "r") as f:
        data = json.load(f)
    if isinstance(data, dict):
        data = data.get("queries", data.get("data", []))
    return data

def call_eval_chat(base_url: str, query: str) -> dict:
    start = time.time()
    try:
        res = requests.post(
            base_url,
            json={"query": query},
            timeout=TIMEOUT_SECONDS,
        )
        latency = round(time.time() - start, 3)
        body = res.json() if res.headers.get("content-type", "").startswith("application/json") else None
        
        return {
            "status_code": res.status_code,
            "latency_seconds": latency,
            "body": body,
            "error": None if res.status_code == 200 else f"HTTP {res.status_code}"
        }
    except Exception as e:
        return {
            "status_code": None,
            "latency_seconds": round(time.time() - start, 3),
            "body": None,
            "error": str(e)
        }

def run():
    dataset = load_dataset(DATASET_PATH)
    if LIMIT:
        dataset = dataset[:LIMIT]

    results = []
    total = len(dataset)
    print(f"=== Starting Eval Run: {total} queries (with dynamic sleep calibration) ===")

    gen_sleep = 28.0
    non_gen_sleep = 2.0

    for i, row in enumerate(dataset, start=1):
        qid = row.get("query_id", f"UNKNOWN_{i}")
        query_text = row.get("query", "")
        print(f"[{i}/{total}] {qid} -> \"{query_text[:80]}\"")

        call_result = call_eval_chat(BASE_URL, query_text)
        if call_result["error"]:
            print(f"  FAILED: {call_result['error']}", file=sys.stderr)

        record = {
            "query_id": qid,
            "query": query_text,
            "expected_intent": row.get("expected_intent"),
            "expected_outcome": row.get("expected_outcome"),
            "request_status_code": call_result["status_code"],
            "latency_seconds": call_result["latency_seconds"],
            "request_error": call_result["error"],
            "response_body": call_result["body"],
            "gold_review_ids": row.get("gold_review_ids", []),
            "expected_review_ids": row.get("expected_review_ids", []),
            "expected_retrieval_strategy": row.get("expected_retrieval_strategy"),
            "gold_relevant_asins": row.get("gold_relevant_asins", []),
            "expected_grounded_asins": row.get("expected_grounded_asins", []),
            "relevance_scores": row.get("relevance_scores"),
            "seed_asin": row.get("seed_asin"),
        }
        results.append(record)
        
        with open(OUT_PATH, "w") as f:
            json.dump(results, f, indent=2)

        # Dynamic sleep logic
        ran_generation = False
        if isinstance(call_result["body"], dict):
            gen_res = call_result["body"].get("generation_result")
            if gen_res is not None:
                ran_generation = True

        latency = call_result["latency_seconds"]
        if ran_generation:
            # If latency is abnormally high, we hit a rate limit retry. Recalibrate sleep!
            if latency > 4.0:
                gen_sleep = min(gen_sleep + 5.0, 45.0)
                print(f"  [Rate Limit Retry! Latency: {latency}s. Increasing sleep to {gen_sleep}s]")
            else:
                # Slowly decay sleep window if it had been increased
                if gen_sleep > 28.0:
                    gen_sleep = max(gen_sleep - 1.0, 28.0)
            sleep_time = gen_sleep
        else:
            sleep_time = non_gen_sleep

        print(f"  Latency: {latency}s | Gen: {ran_generation} | Sleeping: {sleep_time}s")
        time.sleep(sleep_time)

    succeeded = sum(1 for r in results if r["request_error"] is None and r["request_status_code"] == 200)
    failed = len(results) - succeeded
    print("\n=== Eval Run Complete ===")
    print(f"Total: {len(results)} | Succeeded: {succeeded} | Failed: {failed}")
    print(f"Saved to: {OUT_PATH}")

if __name__ == "__main__":
    run()
