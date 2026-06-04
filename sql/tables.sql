CREATE TABLE IF NOT EXISTS products_table (
  asin              TEXT PRIMARY KEY,
  title             TEXT,
  brand             TEXT,
  category          JSONB,
  price             DOUBLE PRECISION,
  price_raw         TEXT,
  main_cat          TEXT,
  description       JSONB,
  feature           JSONB,
  source_run        TEXT,
  updated_at        TIMESTAMPTZ DEFAULT now(),

  search_vector tsvector GENERATED ALWAYS AS (
    setweight(to_tsvector('english', coalesce(title,'')), 'A') ||

    setweight(to_tsvector('english', coalesce(brand,'')), 'B') ||

    setweight(to_tsvector('english', coalesce(description::text,'')), 'B') ||

    setweight(to_tsvector('english', coalesce(feature::text,'')), 'B') ||

    setweight(to_tsvector('english', coalesce(category::text,'')), 'C') ||

    setweight(to_tsvector('english', coalesce(main_cat,'')), 'C') ||

    setweight(to_tsvector('english', coalesce(price_raw,'')), 'D')
  ) STORED
);

CREATE EXTENSION IF NOT EXISTS pg_trgm;

CREATE INDEX idx_products_title_trgm
ON products_table USING gin(title gin_trgm_ops);

CREATE INDEX idx_products_brand_trgm
ON products_table USING gin(brand gin_trgm_ops);

CREATE INDEX idx_products_search_vector
ON products_table USING gin(search_vector);

CREATE TABLE IF NOT EXISTS reviews_table (
  review_id         TEXT PRIMARY KEY,
  asin              TEXT NOT NULL,
  review_text       TEXT,
  summary_text      TEXT,
  source_run        TEXT,
  updated_at        TIMESTAMPTZ DEFAULT now(),
  embedding_status  BOOLEAN DEFAULT FALSE,
  embedded_at       TIMESTAMPTZ,
  embedding_model   TEXT,

  search_vector tsvector GENERATED ALWAYS AS ( 
    setweight(to_tsvector('english', coalesce(summary_text, '')), 'A') || 
    setweight(to_tsvector('english', coalesce(review_text, '')), 'B') 
  ) STORED
);

CREATE INDEX IF NOT EXISTS idx_reviews_id ON reviews_table(review_id);

CREATE INDEX IF NOT EXISTS idx_reviews_search_vector ON reviews_table USING GIN(search_vector);

CREATE TABLE IF NOT EXISTS rag_ingest_state (
  id                INT PRIMARY KEY DEFAULT 1,
  next_shard_idx    INT NOT NULL DEFAULT 0,
  updated_at        TIMESTAMPTZ DEFAULT now()
);

INSERT INTO rag_ingest_state (id, next_shard_idx)
VALUES (1, 0)
ON CONFLICT (id) DO NOTHING;
