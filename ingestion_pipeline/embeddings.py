"""
Generates and stores embeddings for review texts.

This module fetches unprocessed reviews from a PostgreSQL database,
generates embeddings using SentenceTransformers, and stores the resulting
vectors in a Qdrant vector database.
"""

import logging
import psycopg2
import uuid

import pandas as pd
from pathlib import Path
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer

from config.settings import settings

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('embeddings.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

EMBEDDING_MODEL_NAME = settings.embedding_model
COLLECTION_NAME = "reviews_embeddings"

BATCH_SIZE = 10000

model = SentenceTransformer(EMBEDDING_MODEL_NAME)

def get_connection():
    """
    Establishes and returns a connection to the PostgreSQL database.
    """
    try:
        conn = psycopg2.connect(settings.postgres_url)
        logger.info(f"Successfully connected to PostgreSQL database: {settings.postgres_db}")
        return conn
    except Exception as e:
        logger.error(f"Failed to connect to PostgreSQL: {e}")
        raise


def fetch_reviews(conn):
    """
    Fetches a batch of reviews that have not yet been embedded.
    Combines review and summary texts into a single string for embedding.
    """
    try:
        with conn.cursor() as cur:
            query = """
                SELECT review_id,asin,review_text,summary_text 
                FROM reviews_table WHERE 
                embedding_status = FALSE 
                ORDER BY review_id
                LIMIT %s;
            """
            cur.execute(query, (BATCH_SIZE,))
            rows = cur.fetchall()
        
        df = pd.DataFrame(rows, columns=['review_id', 'asin', 'review_text', 'summary_text'])
        df['text'] = df['review_text'].fillna('') + ' ' + df['summary_text'].fillna('')
        df.drop(columns=['review_text', 'summary_text'], inplace=True)
        logger.info(f"Fetched {len(df)} reviews for embedding")
        return df
    except Exception as e:
        logger.error(f"Error fetching reviews: {e}")
        raise
    

def mark_embeddings_complete(conn, model: str, review_ids: list[str]):
    """
    Updates the database to mark a list of reviews as successfully embedded.
    Records the timestamp and the embedding model used.
    """
    try:
        with conn.cursor() as cur:
            query = """
                UPDATE reviews_table 
                SET embedding_status = TRUE, embedded_at = NOW(), embedding_model = %s
                WHERE review_id = ANY(%s);
            """
            cur.execute(query, (model, review_ids))
        logger.debug(f"Marked {len(review_ids)} embeddings as complete")
    except Exception as e:
        logger.error(f"Error marking embeddings complete: {e}")
        raise
    


def create_embeddings(conn, df, qdrant_batch_size:int = 256):
    """
    Generates embeddings for review texts and upserts them to Qdrant.
    Also updates the PostgreSQL database to mark them as completed.
    """
    try:
        logger.info(f"Starting embedding creation for {len(df)} reviews")
        client = QdrantClient(url=settings.qdrant_url)
        logger.info(f"Connected to Qdrant at {settings.qdrant_host}:{settings.qdrant_port}")
        
        existing_names = [c.name for c in client.get_collections().collections]
        if COLLECTION_NAME not in existing_names:
            logger.info(f"Creating new Qdrant collection: {COLLECTION_NAME}")
            client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(size=384, distance=Distance.COSINE)
            )
        else:
            logger.info(f"Using existing Qdrant collection: {COLLECTION_NAME}")

        texts = df["text"].fillna("").tolist()
        formatted_texts = [
            f"Represent this sentence for retrieval: {t}"
            for t in texts
        ]
        
        logger.info(f"Encoding {len(formatted_texts)} texts using {EMBEDDING_MODEL_NAME}")
        embeddings = model.encode(
            formatted_texts, batch_size=256, normalize_embeddings=True, show_progress_bar=True
        )
        logger.info(f"Successfully encoded {len(embeddings)} embeddings")

        rows = df.to_dict("records")

        points = [
            PointStruct(
                id = uuid.uuid5(uuid.NAMESPACE_OID, row["review_id"]), 
                vector=emb,
                payload={
                    "review_id": row["review_id"],
                    "asin": row["asin"],
                    "text": row["text"],
                },
            )
            for row, emb in zip(rows, embeddings)
        ]
        logger.info(f"Created {len(points)} point structures")

        for i in range(0, len(points), qdrant_batch_size):
            batch_points=points[i:i+qdrant_batch_size]
            client.upsert(collection_name=COLLECTION_NAME, points = batch_points )

            batch_review_ids = [p.payload["review_id"] for p in batch_points]
            mark_embeddings_complete(conn, EMBEDDING_MODEL_NAME, batch_review_ids)

            conn.commit()
            logger.info(f"Completed batch {i} - {i + len(batch_points)}")

        logger.info("All embeddings successfully upserted to Qdrant")
    except Exception as e:
        logger.error(f"Error creating embeddings: {e}")
        raise


def main():
    """
    Main execution loop that continuously fetches and processes batches
    of reviews until no unprocessed reviews remain.
    """
    logger.info("Starting embeddings main process")
    conn = get_connection()
    try:
        while True:
            df = fetch_reviews(conn)
            if len(df) == 0:
                logger.info("No rows to embed. Exiting.")
                return
        
            create_embeddings(conn, df)

    except Exception as e:
        logger.error(f"Fatal error in main: {e}")
        conn.rollback()
        raise 
    finally:
        conn.close()
        logger.info("Closed database connection")

if  __name__ == "__main__":
    main()