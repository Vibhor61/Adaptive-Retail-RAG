"""
Fetches and analyzes retrieval-specific latency metrics from Phoenix traces.
Connects to the Phoenix GraphQL API to extract start and end times for spans.
Filters and computes latency statistics for the targeted retrieval pipeline.
Exports the resulting parsed metrics to a CSV file for further inspection.
"""

import requests
import pandas as pd
from config.settings import settings
from datetime import datetime, timezone
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent

PHOENIX_GRAPHQL = f"{settings.phoenix_url_ui}/graphql"
PROJECT_ID = "UHJvamVjdDox"
SPAN_TARGET = "retrieval_pipeline"

START_TIME = "2026-06-26 03:50:00"
START_DT = datetime.strptime(START_TIME, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)


def fetch_retrieval_spans(project_id: str, limit: int = 1000):
    """
    Fetches recent retrieval pipeline spans via Phoenix GraphQL.
    Executes a query to retrieve span details like start time and latency.
    Returns a list of parsed span nodes containing the required data.
    """
    query = """
    query GetSpans($projectId: ID!, $first: Int!, $filterCondition: String) {
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
                trace {
                  traceId
                }
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
                "first": limit,
                "filterCondition": f'name == "{SPAN_TARGET}"',
            }
        },
    )
    resp.raise_for_status()
    result = resp.json()
    if "errors" in result:
        raise RuntimeError(f"GraphQL errors: {result['errors']}")

    edges = result["data"]["node"]["spans"]["edges"]
    return [e["node"] for e in edges]



def parse_iso(dt_str):
    if dt_str is None:
        return None
    return datetime.fromisoformat(dt_str.replace("Z", "+00:00"))


def extract_retrieval_latency(spans):
    """
    Filters target spans and returns a DataFrame containing latency metrics.
    Computes latency from timestamps if not explicitly provided in the span.
    Returns a pandas DataFrame sorted by the execution start time.
    """
    rows = []

    for span in spans:
        if span["name"] != SPAN_TARGET:
            continue

        start = parse_iso(span["startTime"])
        if start is None or start < START_DT:
            continue

        end = parse_iso(span["endTime"])

        if span.get("latencyMs") is not None:
            latency_ms = span["latencyMs"]
        elif end is not None:
            latency_ms = (end - start).total_seconds() * 1000
        else:
            latency_ms = None

        rows.append({
            "trace_id": span["trace"]["traceId"],
            "start_time": start,
            "end_time": end,
            "latency_ms": latency_ms,
        })

    return pd.DataFrame(rows)


def main():
    """
    Fetches retrieval latency data, prints basic stats, and exports to CSV.
    Acts as the main execution flow for querying and summarizing latency.
    Outputs the results locally to the script's directory.
    """
    spans = fetch_retrieval_spans(project_id=PROJECT_ID, limit=1000)
    df = extract_retrieval_latency(spans)

    if df.empty:
        print("No matching spans found.")
        return

    print("Stats (latency_ms):\n", df["latency_ms"].describe())

    out_file = SCRIPT_DIR / "data" / "retrieval_node_latency.csv"
    df.sort_values("start_time").to_csv(out_file, index=False)


if __name__ == "__main__":
    main()