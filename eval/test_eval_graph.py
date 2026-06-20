import requests
import json
import time

BASE_URL = "http://localhost:8000/eval_chat"

TEST_QUERIES = [
    # {
    #     "name": "SPEC_LOOKUP",
    #     "query": "What are the specs of DevilCase HTC Desire EYE Aluminum Alloy Protective Bumper?"
    # },
    # {
    #     "name": "EXPERIENTIAL",
    #     "query": "How well do OtterBox phone accessories protect devices against catastrophic drops according to customer reviews?"
    # },
    # {
    #     "name": "COMPARISON",
    #     "query": "What features are common across MIcrosoft Lumia cases listed in the store?"
    # },
    {
        
        "name": "RECOMMENDATION",
        "query": "Recommend a phone case phone that survived multiple drops."
    }
]

def call(payload):
    start = time.time()
    res = requests.post(BASE_URL, json=payload, timeout=120)
    latency = time.time() - start

    return {
        "status": res.status_code,
        "latency": round(latency, 3),
        "body": res.json() if res.headers.get("content-type", "").startswith("application/json") else None
    }

def summarize(result):
    body = result["body"] or {}
    control = body.get("control", {})

    return {
        "status": result["status"],
        "latency": result["latency"],
        "answer": body.get("answer"),
        "citations": body.get("citations", []),
        "failed": control.get("should_clarify", False),
        "failure_stage": control.get("failure_stage"),
        "failure_reason": control.get("failure_reason"),
        
        # --- Intermediate Debug State ---
        "router_result": body.get("router_result"),
        "router_guardrails": body.get("router_guardrails"),
        "retrieval_result": body.get("retrieval_result"),
        "retrieval_guardrails": body.get("retrieval_guardrails"),
        "generation_result": body.get("generation_result")
    }

def run():
    print("\n=== EVAL START (FULL SMOKE TEST) ===\n")

    summary_rows = []

    for q in TEST_QUERIES:
        print(f"\n==================================================")
        print(f"[{q['name']}] -> Query: {q['query']}")
        print(f"==================================================")

        result = call({"query": q["query"]})
        summary = summarize(result)

        # Print the detailed intermediate results for raw visibility
        print(json.dumps(summary, indent=2))

        summary_rows.append({
            "name": q["name"],
            "status": summary["status"],
            "latency": summary["latency"],
            "failed": summary["failed"],
            "failure_stage": summary["failure_stage"]
        })

    print("\n=== FINAL EXECUTION SUMMARY ===\n")
    print(f"{'NAME':<15} {'STATUS':<8} {'LATENCY':<10} {'FAILED':<8} {'FAILURE_STAGE'}")
    print("-" * 70)

    for row in summary_rows:
        print(
            f"{row['name']:<15} "
            f"{row['status']:<8} "
            f"{row['latency']:<10} "
            f"{str(row['failed']):<8} "
            f"{str(row['failure_stage'])}"
        )

    print("\n=== DONE ===\n")

if __name__ == "__main__":
    run()