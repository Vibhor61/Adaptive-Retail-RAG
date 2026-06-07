from orchestration.controller import controller

test_queries = [
    # "What are the specs of DevilCase HTC Desire EYE Aluminum Alloy Protective Bumper?",
    # "AirPods Pro vs Sony WH-1000XM5 for gym",
    "best wireless earbuds under 5000",
    # "track my order",
]

if __name__ == "__main__":
    for query in test_queries:
        print(f"\n{'='*60}")
        print(f"QUERY: {query}")
        print('='*60)
        
        result = controller(query)
        
        if result.system_failure:
            print(f"FAILED at stage: {result.stage}")
            print(f"Reason: {result.system_failure}")
        else:
            print(f"ANSWER:\n{result.answer}")
            print(f"\nCITATIONS: {len(result.citations)}")
            for c in result.citations:
                print(f"  [{c.citation_id}] ASIN={c.asin} | {c.evidence_text[:80]}...")