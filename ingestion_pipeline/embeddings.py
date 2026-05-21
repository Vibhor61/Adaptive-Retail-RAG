import pandas as pd
import os
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer
import psycopg2
import uuid

DB_CONFIG = {
    "host" : os.getenv("POSTGRES_HOST"),
    "database" : os.getenv("POSTGRES_DB"),
    "user" : os.getenv("POSTGRES_USER"),
    "password" : os.getenv("POSTGRES_PASSWORD"),
    "port" : int(os.getenv("POSTGRES_PORT"))
}

QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))

EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"
COLLECTION_NAME = "reviews_embeddings"

BATCH_SIZE = 10000

def get_connection():
    return psycopg2.connect(**DB_CONFIG)


def fetch_reviews(conn):
    with conn.cursor() as cur:
        query = """
            SELECT review_id,asin,review_text,summary_text 
            FROM reviews_table WHERE 
            embedding_status = FALSE AND
                (review_text IS NOT NULL OR summary_text IS NOT NULL)
            LIMIT %s;
        """
        cur.execute(query, (BATCH_SIZE,))
        rows = cur.fetchall()
        
    df = pd.DataFrame(rows, columns=['review_id', 'asin', 'review_text', 'summary_text'])
    df['text'] = df['review_text'].fillna('') + ' ' + df['summary_text'].fillna('')
    df.drop(columns=['review_text', 'summary_text'], inplace=True)
    return df
    

def mark_embeddings_complete(conn, model:str, review_ids: list[str]):
    with conn.cursor() as cur:
        query = """
            UPDATE reviews_table 
            SET embedding_status = TRUE, embedded_at = NOW(), embedding_model = %s
            WHERE review_id = ANY(%s);
        """

        cur.execute(query, (model,review_ids))
    


def create_embeddings(conn, df, qdrant_batch_size:int = 256):

    client = QdrantClient(host=QDRANT_HOST,port=QDRANT_PORT)
    existing_names = [c.name for c in client.get_collections().collections]
    if COLLECTION_NAME not in existing_names:
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=384, distance=Distance.COSINE)
        )

    model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    texts = df["text"].fillna("").tolist()
    formatted_texts = [
        f"Represent this sentence for retrieval: {t}"
        for t in texts
    ]
    
    embeddings = model.encode(
        formatted_texts, normalize_embeddings=True, show_progress_bar=True
    )

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

    for i in range(0, len(points), qdrant_batch_size):

        batch_points=points[i:i+qdrant_batch_size]
        client.upsert(collection_name=COLLECTION_NAME, points = batch_points )

        batch_review_ids = [p.payload["review_id"] for p in batch_points]
        mark_embeddings_complete(conn, EMBEDDING_MODEL_NAME, batch_review_ids)

        conn.commit()
        print(f"Completed batch {i} - {i + len(batch_points)}")

    print("Embeddings upserted to Qdrant")


def main():
    conn = get_connection()
    try:
        while True:
            df = fetch_reviews(conn)
            if len(df) == 0:
                print("No rows to embed. Exiting.")
                return
        
            print(f"Fetched {len(df)} reviews")
            create_embeddings(conn, df)

    except Exception as e:
        conn.rollback()
        raise ("Error during embeddings", e)
    finally:
        conn.close()

if  __name__ == "__main__":
    main()