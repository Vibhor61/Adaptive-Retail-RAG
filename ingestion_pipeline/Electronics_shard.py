"""
Script to shard the raw Amazon reviews dataset into smaller chunks.

This module reads a compressed JSON file of electronics reviews, extracts
relevant fields using helper functions, and writes them out into smaller,
compressed JSON lines files (shards) for easier processing.
"""

import gzip
import json
from utility_functions.ingestion_helper import iter_rows, extract_reviews
import os

INPUT_FILE = "../Data/raw_data/Cell_Phones_and_Accessories.json.gz"
OUTPUT_PREFIX = "../Data/shards"
SHARD_SIZE = 100000

def shard_reviews():
    """
    Iterates through raw reviews, extracts relevant data, and writes to sharded files.
    Creates a new compressed file whenever the current shard reaches SHARD_SIZE.
    """
    shard_id = 0
    count = 0 
    total = 0

    os.makedirs(OUTPUT_PREFIX, exist_ok=True)
    output = gzip.open(f"{OUTPUT_PREFIX}/shard_{shard_id:03d}.jsonl.gz","wt")

    for row in iter_rows(INPUT_FILE):

        raw = extract_reviews(row)
        if raw is None:
            continue
        
        output.write(json.dumps(raw, ensure_ascii=False) + "\n")
        count += 1
        total += 1

        if count >= SHARD_SIZE:
            output.close()
            shard_id += 1
            count = 0
            output = gzip.open(f"{OUTPUT_PREFIX}/shard_{shard_id:03d}.jsonl.gz","wt")


    output.close()
    print(f"Total reviews extracted: {total}")


if __name__ == "__main__":
    shard_reviews()