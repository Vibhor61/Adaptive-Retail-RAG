import sys
import time
import os
import statistics
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from routing_layer.entity_resolver import DBEntityLoader

def run_database_resolver_test():
    try:
        loader = DBEntityLoader()
    except Exception as e:
        print(f"Failed to initialize DBEntityLoader connection pool: {e}")
        return

    dataset_path = os.path.join(CURRENT_DIR, "data", "resolver_dataset.txt")
    if not os.path.exists(dataset_path):
        print(f"Failed to locate target dataset file at: {dataset_path}")
        return

    with open(dataset_path, "r", encoding="utf-8") as f:
        test_cases = [line.strip() for line in f if line.strip()]
        
    latencies = []
    total_cases = len(test_cases)
    found_count = 0
    overlap_pass_count = 0

    print("=" * 140)
    print(f"{'INDEX':<6} {'TEST ENTITY':<45} {'FOUND':<8} {'TOP MATCH TYPE':<16} {'TOP SCORE':<12} {'OVERLAP 80%':<12} {'LATENCY':>12}")
    print("=" * 140)

    for idx, entity in enumerate(test_cases, start=1):
        t0 = time.perf_counter()
        
        try:
            candidates = loader.candidate_search(entity)
            elapsed_ms = (time.perf_counter() - t0) * 1000
            latencies.append(elapsed_ms)
            
            num_found = len(candidates)
            if num_found > 0:
                found_count += 1
                top_candidate = candidates[0]
                status = "YES"
                top_type = top_candidate.match_type.value
                top_score = f"{top_candidate.retrieval_score:.4f}"
                
                # Tokenize into clean sets
                gold_tokens = set(entity.lower().split())
                pred_tokens = set(top_candidate.title.lower().split())
                
                # Calculate Bidirectional Overall Overlap (Intersection Over Union)
                intersection = gold_tokens.intersection(pred_tokens)
                union = gold_tokens.union(pred_tokens)
                
                overlap_ratio = len(intersection) / len(union) if union else 0.0
                
                # Check if the overall global title matches by 80%+
                if overlap_ratio >= 0.80:
                    overlap_passed = "PASS"
                    overlap_pass_count += 1
                else:
                    overlap_passed = "FAIL"
            else:
                status = "NO"
                top_type = "none"
                top_score = "0.0000"
                overlap_passed = "FAIL"

            preview = entity[:42] + "..." if len(entity) > 42 else entity
            print(
                f"#{idx:<5} "
                f"{preview:<45} "
                f"{status:<8} "
                f"{top_type:<16} "
                f"{top_score:<12} "
                f"{overlap_passed:<12} "
                f"{elapsed_ms:>11.1f}ms"
            )

            if num_found > 0:
                for rank, c in enumerate(candidates[:2], start=1):
                    trimmed_title = c.title[:75] + "..." if len(c.title) > 75 else c.title
                    print(f"    [{rank}] ASIN: {c.asin or 'N/A':<12} | Score: {c.retrieval_score:.4f} | {trimmed_title}")
                print("-" * 140)

        except Exception as e:
            preview = entity[:42] + "..." if len(entity) > 42 else entity
            print(f"#{idx:<5} {preview:<45} CRASHED! Error: {e}")
            print("-" * 140)

    print("=" * 140)
    print("FINAL EVALUATION METRICS")
    print("=" * 140)
    print(f"Total Test Cases      : {total_cases}")
    print(f"Database Hit Rate     : {found_count}/{total_cases} ({found_count / total_cases * 100:.1f}%)")
    print(f"Token Overlap (80%+)  : {overlap_pass_count}/{total_cases} ({overlap_pass_count / total_cases * 100:.1f}%)")
    
    if latencies:
        mean_lat = statistics.mean(latencies)
        med_lat = statistics.median(latencies)
        p95_lat = sorted(latencies)[int(len(latencies) * 0.95)]
        print(f"Latency Profile       : mean={mean_lat:.1f}ms | median={med_lat:.1f}ms | p95={p95_lat:.1f}ms (n={len(latencies)})")
    print("=" * 140)

if __name__ == "__main__":
    run_database_resolver_test()