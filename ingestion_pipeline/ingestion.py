"""
Loads product metadata and reviews into a PostgreSQL database.

This module provides functions to parse raw JSON data and insert or update
records in the database. It handles both product metadata and chunked review
shards, managing state and avoiding duplicates via batch operations.
"""

import json
import psycopg2
import argparse
import logging
import os 

from psycopg2.extras import execute_values
from pathlib import Path
from config.settings import settings

from utility_functions.ingestion_helper import iter_rows, extract_metadata, extract_reviews

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ingestion.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_PRODUCTS = PROJECT_ROOT / "Data" / "raw_data" / "meta_Cell_Phones_and_Accessories.json.gz"
DEFAULT_REVIEWS = PROJECT_ROOT / "Data" / "shards"

# Products Metadata can change with time we need latest data per ASIN so doing upsert 
# Delete and insert is load heavy
PRODUCT_UPSERT_SQL = '''
    INSERT INTO products_table (
        asin, title, brand, category, price, price_raw, main_cat, description, feature, source_run
    )
    VALUES %s
    ON CONFLICT (asin)
    DO UPDATE SET
        title = EXCLUDED.title,
        brand = EXCLUDED.brand,
        category = EXCLUDED.category,
        price = EXCLUDED.price,
        price_raw = EXCLUDED.price_raw,
        main_cat = EXCLUDED.main_cat,
        description = EXCLUDED.description,
        feature = EXCLUDED.feature,
        source_run = EXCLUDED.source_run,
        updated_at = now();
'''

# Reviews are immutable
REVIEW_UPSERT_SQL = '''
    INSERT INTO reviews_table (
        review_id, asin, review_text, summary_text, source_run
        )
    VALUES %s
    ON CONFLICT (review_id)
    DO NOTHING;
'''

def load_products(cur, product_file: str, run_date: str, batch_size: int = 5000):
    """
    Reads product metadata from a file and inserts or updates it in the database.
    Returns a dictionary with statistics on rows seen, written, and skipped.
    """
    logger.info(f"Starting product load from {product_file}")
    buffer = []
    seen , written , skipped = 0, 0, 0

    seen_asin = set()
    try:
        for obj in iter_rows(product_file):
            seen += 1
            res = extract_metadata(obj)
            if res is None:
                skipped += 1
                continue
            
            asin, meta = res

            if asin in seen_asin:
                skipped += 1
                continue
            
            seen_asin.add(asin)

            buffer.append((
                asin,
                meta.get("title", ""),
                meta.get("brand", ""),
                json.dumps(meta.get("category", [])),
                meta.get("price", None),
                str(meta["price_raw"]) if meta.get("price_raw") is not None else None,
                meta.get("main_cat", ""),
                json.dumps(meta.get("description", [])),
                json.dumps(meta.get("feature", [])),
                run_date
            ))

            if len(buffer) >= batch_size:
                # Template use due to execute values not supporting jsonb type directly
                execute_values(cur, PRODUCT_UPSERT_SQL, buffer, template="(%s, %s, %s, %s::jsonb, %s, %s, %s, %s::jsonb, %s::jsonb, %s)", page_size=batch_size)
                written += len(buffer)
                logger.debug(f"Inserted {written} products so far")
                buffer.clear()

        if buffer:
            execute_values(cur, PRODUCT_UPSERT_SQL, buffer, template="(%s, %s, %s, %s::jsonb, %s, %s, %s, %s::jsonb, %s::jsonb, %s)", page_size=len(buffer))
            written += len(buffer)

        logger.info(f"Product load complete - Seen: {seen}, Written: {written}, Skipped: {skipped}")
        return {"seen": seen, "written": written, "skipped": skipped}
    except Exception as e:
        logger.error(f"Error loading products: {e}")
        raise


def load_reviews(cur, product_file: str, run_date: str, batch_size: int = 5000):
    """
    Reads review shards from a file and inserts them into the database.
    Returns a dictionary with statistics on rows seen, written, and skipped.
    """
    logger.info(f"Starting review load from {product_file}")
    buffer = []
    seen , written , skipped = 0, 0, 0

    try:
        for obj in iter_rows(product_file):
            seen += 1
            res = extract_reviews(obj)
            if res is None:
                skipped += 1
                continue

            buffer.append((
                res["review_id"],
                res["asin"],
                res.get("review_text", ""),
                res.get("summary_text", ""),
                run_date
            ))

            if len(buffer) >= batch_size:
                execute_values(cur, REVIEW_UPSERT_SQL, buffer, page_size=batch_size)
                written += len(buffer)
                logger.debug(f"Inserted {written} reviews so far")
                buffer.clear()

        if buffer:
            execute_values(cur, REVIEW_UPSERT_SQL, buffer, page_size=len(buffer))
            written += len(buffer)

        logger.info(f"Review load complete - Seen: {seen}, Written: {written}, Skipped: {skipped}")
        return {"seen": seen, "written": written, "skipped": skipped}
    except Exception as e:
        logger.error(f"Error loading reviews: {e}")
        raise


def update_rag_ingest_state(cur, shard_idx: int):
    """
    Updates the ingestion state table to record the next shard to process.
    """
    try:
        cur.execute('''
            INSERT INTO rag_ingest_state (id, next_shard_idx)
            VALUES (1, %s)
            ON CONFLICT (id) 
            DO UPDATE SET next_shard_idx = EXCLUDED.next_shard_idx, updated_at = now()
        ''', (shard_idx+1,))
        logger.debug(f"Updated ingest state to shard {shard_idx+1}")
    except Exception as e:
        logger.error(f"Error updating ingest state: {e}")
        raise
        

def run_loader(product_file_path:str, review_file_path:str, run_date:str, start_shard:int, end_shard:int, overwrite_partition:bool = False, metadata:bool=False):
    """
    Main orchestration function to run the data loader.
    Optionally clears existing partitions, loads product metadata, and iterates over review shards.
    """
    logger.info(f"Starting data loader - Run date: {run_date}, Shards: {start_shard}-{end_shard}")
    logger.debug(f"Database config: {settings.postgres_url}")
    
    conn = psycopg2.connect(settings.postgres_url)
    try:
        with conn.cursor() as cur:
            if overwrite_partition:
                logger.info(f"Overwriting partition for run_date: {run_date}")
                cur.execute("DELETE FROM reviews_table WHERE source_run = %s;", (run_date,))
                cur.execute("DELETE FROM products_table WHERE source_run = %s;", (run_date,))
            
            prod_stats = None
            if metadata:
                logger.info("Loading product metadata")
                prod_stats = load_products(cur, product_file_path, run_date)

            all_reviews_stats = {"seen": 0, "written": 0, "skipped": 0}
            logger.info(f"Processing shards {start_shard} to {end_shard}")
            for shard_idx in range(start_shard, end_shard+1):
                shard_file = os.path.join(review_file_path, f"shard_{shard_idx:03d}.jsonl.gz")
                logger.info(f"Loading shard {shard_idx}: {shard_file}")
                rev_stats = load_reviews(cur, shard_file, run_date)
                update_rag_ingest_state(cur, shard_idx)
                logger.info(f"Shard {shard_idx} stats - {rev_stats}")

                all_reviews_stats["seen"] += rev_stats["seen"]
                all_reviews_stats["written"] += rev_stats["written"]
                all_reviews_stats["skipped"] += rev_stats["skipped"]

            conn.commit()
            logger.info(f"Data loader completed successfully. Total reviews - Seen: {all_reviews_stats['seen']}, Written: {all_reviews_stats['written']}, Skipped: {all_reviews_stats['skipped']}")
        return {"products": prod_stats, "reviews": all_reviews_stats}
    
    except Exception as e:
        logger.error(f"Error during loading: {e}")
        conn.rollback()
        raise
    finally:
        conn.close()
        logger.info("Closed database connection")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Loading Amazon data into Postgres")
    
    parser.add_argument(
        "--products",
        default = str(DEFAULT_PRODUCTS),
        help="Path to metadata file"
    )
    
    parser.add_argument(
        "--reviews",
        default = str(DEFAULT_REVIEWS),
        help="Path to review file"
    )
    
    parser.add_argument(
        "--run-date",
        required=True,
        help="Run date partition (YYYY-MM-DD)"
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite partition for run_date"
    )

    parser.add_argument(
        "--metadata",
        action="store_true",
        help="If needed to reload metadata"
    )

    parser.add_argument(
        "--starting-shard",
        type=int,
        required=True,
        help="Starting shard number"
    )

    parser.add_argument(
        "--ending-shard",
        type=int,
        required=True,
        help="Ending shard number "
    )

    args = parser.parse_args()

    logger.info(f"Ingestion script started with arguments: {vars(args)}")
    try:
        stats = run_loader(
            product_file_path=args.products,
            review_file_path=args.reviews,
            run_date=args.run_date,
            overwrite_partition=args.overwrite,
            metadata=args.metadata,
            start_shard=args.starting_shard,
            end_shard=args.ending_shard
        )
        logger.info(f"Final statistics: {stats}")
        print(stats)
    except Exception as e:
        logger.error(f"Ingestion script failed with error: {e}")
        raise