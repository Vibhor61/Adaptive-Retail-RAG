import re
import html
import psycopg2
from routing_layer.entity_resolver import DB_CONFIG

# Your target Windows output path
OUTPUT_PATH = r"C:\Users\risha\OneDrive\Desktop\RAG\output.txt"

def clean_title(title: str) -> str:
    title = html.unescape(title)                         # &amp; → &
    title = re.sub(r'\(TM\)|\(R\)', '', title)           # remove TM/R
    title = re.sub(r'Non-Retail Packaging', '', title, flags=re.IGNORECASE)
    title = re.sub(r'\s*-\s*$', '', title)               # trailing dash
    title = re.sub(r'\[.*?\]', '', title)                # remove [...] blocks
    title = re.sub(r'\((?![^)]*\d)[^)]*\)', '', title)  # remove non-spec parentheticals
    title = re.sub(r'\s+', ' ', title).strip()
    return title

def main():
    print(r"Connecting to DB and exporting cleaned titles to Desktop...")
    
    # Connect and fetch data
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()
    
    # Let's up the limit to 50 as you wanted earlier
    cur.execute("SELECT title FROM products_table ORDER BY RANDOM() LIMIT 50")
    rows = cur.fetchall()
    
    # Clean and write directly to file
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for row in rows:
            if row[0]: # Ensure title is not null
                cleaned = clean_title(row[0])
                f.write(cleaned + "\n")
                
    cur.close()
    conn.close()
    
    print(f"Successfully wrote 50 cleaned titles to: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()