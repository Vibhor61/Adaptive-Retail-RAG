"""
Fetches and analyzes overall pipeline latency data from Phoenix traces.
It calculates latency percentiles, determines interquartile ranges,
and identifies execution outliers for both overall and retrieval pipelines.
The script finally exports the cleaned and outlier trace data to CSV files.
"""

import requests
import pandas as pd
from datetime import datetime, timezone
from pathlib import Path
from config.settings import settings

START_DT = datetime(2026, 6, 26, 3, 50, 0, tzinfo=timezone.utc)

SCRIPT_DIR = Path(__file__).resolve().parent

PHOENIX_GRAPHQL = f"{settings.phoenix_url_ui}/graphql"
PROJECT_ID = "UHJvamVjdDox"
LATEST_N = 1000


def fetch_spans_by_name(project_id: str, span_name: str, n: int = 200):
    """
    Fetches a specific number of latest spans by name using Phoenix GraphQL.
    It takes the project ID and span name, executes a POST request,
    and returns a list of matched span nodes.
    """
    query = """
    query GetLatestSpans($projectId: ID!, $first: Int!, $filterCondition: String) {
      node(id: $projectId) {
        ... on Project {
          spans(
            first: $first,
            sort: { col: startTime, dir: desc },
            filterCondition: $filterCondition
          ) {
            edges {
              node {
                name
                startTime
                endTime
                latencyMs
                trace { traceId }
              }
            }
          }
        }
      }
    }
    """
    resp = requests.post(
        PHOENIX_GRAPHQL,
        json={
            "query": query,
            "variables": {
                "projectId": project_id,
                "first": n,
                "filterCondition": f'name == "{span_name}"',
            },
        },
    )
    resp.raise_for_status()
    result = resp.json()
    if "errors" in result:
        raise RuntimeError(f"GraphQL errors: {result['errors']}")
    return [e["node"] for e in result["data"]["node"]["spans"]["edges"]]


def parse_iso(dt_str):
    """
    Parses an ISO datetime string into a timezone-aware datetime object.
    Returns None if the input string is None.
    """
    if dt_str is None:
        return None
    return datetime.fromisoformat(dt_str.replace("Z", "+00:00"))


def build_df(spans):
    """
    Constructs a pandas DataFrame from raw span data, computing latencies.
    It processes start and end times, calculates latency in milliseconds and seconds,
    and returns the structured data as a pandas DataFrame.
    """
    rows = []
    for s in spans:
        start = parse_iso(s["startTime"])
        if start is None or start < START_DT:
            continue
        end = parse_iso(s["endTime"])
        lat = s.get("latencyMs")
        if lat is None and end and start:
            lat = (end - start).total_seconds() * 1000
        rows.append({
            "trace_id": s["trace"]["traceId"],
            "start_time": start,
            "end_time": end,
            "latency_ms": lat,
            "latency_s": round(lat / 1000, 3) if lat is not None else None,
        })
    return pd.DataFrame(rows)


def iqr_bounds(series):
    """
    Calculates the lower and upper bounds for outlier detection using the IQR method.
    Returns a tuple containing the lower and upper bound thresholds.
    """
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    return q1 - 1.5 * iqr, q3 + 1.5 * iqr


def print_stats(df, label):
    """
    Prints formatted summary statistics for latencies including percentiles.
    It formats count, mean, standard deviation, and multiple percentiles
    into a readable table output for the given dataframe.
    """
    sep = "=" * 66
    print(f"\n{sep}")
    print(f"  {label}  (n={len(df)})")
    print(sep)
    print(f"  {'Metric':<18} {'ms':>12}  {'seconds':>10}")
    print(f"  {'-'*18} {'-'*12}  {'-'*10}")

    pcts = [0.25, 0.5, 0.75, 0.90, 0.95, 0.99]
    desc = df["latency_ms"].describe(percentiles=pcts)
    labels_map = {
        "count": "Count", "mean": "Mean", "std": "Std Dev",
        "min": "Min", "25%": "p25", "50%": "Median (p50)",
        "75%": "p75", "90%": "p90", "95%": "p95", "99%": "p99", "max": "Max"
    }
    for k, v in desc.items():
        lbl = labels_map.get(k, k)
        if k == "count":
            print(f"  {lbl:<18} {int(v):>12}  {'':>10}")
        else:
            print(f"  {lbl:<18} {v:>12.1f}  {v/1000:>10.3f}s")


def print_outliers(outliers, label):
    """
    Prints a formatted table of traces identified as latency outliers.
    It displays the trace ID, start time, latency in milliseconds and seconds,
    as well as the z-score for each outlier trace.
    """
    sep = "=" * 66
    print(f"\n{sep}")
    print(f"  OUTLIERS — {label}  ({len(outliers)} traces)")
    print(sep)
    if outliers.empty:
        print("  None detected.")
        return
    print(f"  {'#':<4} {'trace_id':<36}  {'start (UTC)':<22}  {'ms':>10}  {'sec':>8}  {'z':>6}")
    print(f"  {'-'*4} {'-'*36}  {'-'*22}  {'-'*10}  {'-'*8}  {'-'*6}")
    for i, (_, row) in enumerate(outliers.iterrows(), 1):
        ts = str(row["start_time"])[:22]
        print(f"  {i:<4} {row['trace_id']:<36}  {ts:<22}  {row['latency_ms']:>10.0f}  {row['latency_ms']/1000:>8.3f}s  {row['z_score']:>6.2f}")


def main():
    """
    Fetches latency data, identifies outliers, prints stats, and exports CSVs.
    It processes both overall and retrieval pipeline spans, filters out outliers,
    displays performance statistics, and writes the results to local files.
    """
    overall_spans = fetch_spans_by_name(PROJECT_ID, "eval_pipeline", n=140)
    df_overall = build_df(overall_spans)

    lo, hi = iqr_bounds(df_overall["latency_ms"])
    df_overall["z_score"] = (
        (df_overall["latency_ms"] - df_overall["latency_ms"].mean())
        / df_overall["latency_ms"].std()
    )
    df_overall["is_outlier"] = (df_overall["latency_ms"] < lo) | (df_overall["latency_ms"] > hi)

    outliers_overall = df_overall[df_overall["is_outlier"]].sort_values("latency_ms", ascending=False)
    df_clean = df_overall[~df_overall["is_outlier"]].copy()

    clean_trace_ids = set(df_clean["trace_id"])

    print_outliers(outliers_overall, "eval_pipeline (IQR upper fence = {:.0f} ms)".format(hi))
    print_stats(df_clean, "OVERALL LATENCY — eval_pipeline (outliers removed)")

    retrieval_spans = fetch_spans_by_name(PROJECT_ID, "retrieval_pipeline", n=1000)
    df_ret = build_df(retrieval_spans)

    df_ret_matched = df_ret[df_ret["trace_id"].isin(clean_trace_ids)].copy()

    if not df_ret_matched.empty:
        lo_r, hi_r = iqr_bounds(df_ret_matched["latency_ms"])
        df_ret_matched["z_score"] = (
            (df_ret_matched["latency_ms"] - df_ret_matched["latency_ms"].mean())
            / df_ret_matched["latency_ms"].std()
        )
        df_ret_matched["is_outlier"] = (
            (df_ret_matched["latency_ms"] < lo_r) | (df_ret_matched["latency_ms"] > hi_r)
        )
        outliers_ret = df_ret_matched[df_ret_matched["is_outlier"]].sort_values("latency_ms", ascending=False)
        df_ret_clean = df_ret_matched[~df_ret_matched["is_outlier"]].copy()

        print_outliers(outliers_ret, "retrieval_pipeline (IQR upper fence = {:.0f} ms)".format(hi_r))
        print_stats(df_ret_matched, "RETRIEVAL LATENCY — retrieval_pipeline (all matched)")
        print_stats(df_ret_clean,   "RETRIEVAL LATENCY — retrieval_pipeline (outliers removed)")

    df_clean.to_csv(SCRIPT_DIR / "data" / "overall_latency_clean.csv", index=False)
    outliers_overall.to_csv(SCRIPT_DIR / "data" / "overall_latency_outliers.csv", index=False)
    if not df_ret_matched.empty:
        df_ret_matched.to_csv(SCRIPT_DIR / "data" / "retrieval_latency_200.csv", index=False)
        outliers_ret.to_csv(SCRIPT_DIR / "data" / "retrieval_latency_outliers.csv", index=False)


if __name__ == "__main__":
    main()
