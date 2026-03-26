"""
Generate embeddings WITHOUT chunking.
Each paper → one embedding vector from: title + abstract
Output: embeddings saved as .npy + metadata as .json
"""

import json
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import os




PROJECT_DIR = "/content/drive/MyDrive/arxiv-rag-project"
RESULTS_DIR = os.path.join(PROJECT_DIR, "results")

os.makedirs(RESULTS_DIR, exist_ok=True)

INPUT_FILE  = os.path.join(PROJECT_DIR, "cs_papers_100k_with_institutions_final1.json")
OUTPUT_EMB  = os.path.join(RESULTS_DIR, "embeddings_no_chunk.npy")
OUTPUT_META = os.path.join(RESULTS_DIR, "embeddings_meta_no_chunk.json")

BATCH_SIZE  = 256
MODEL_NAME  = "sentence-transformers/all-MiniLM-L6-v2"




def build_text(row: dict) -> str:
    """Combine title + abstract into one string per paper."""
    title    = (row.get("title")    or "").strip().replace("\n", " ")
    abstract = (row.get("abstract") or "").strip().replace("\n", " ")
    return f"{title}. {abstract}"



def main():
    # Load data
    print(f"Loading {INPUT_FILE} ...")
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        records = json.load(f)

    df = pd.DataFrame(records)
    print(f"Total papers loaded: {len(df)}")

    # text corpus
    texts = [build_text(row) for row in records]

    for i, t in enumerate(texts[:2]):
        print(f"\n[Sample {i}] {t[:200]} ...")

    # Load model
    print(f"\nLoading model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)

    # Generate embeddings
    print(f"\nGenerating embeddings (batch_size={BATCH_SIZE}) ...")
    embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True
    )

    print(f"\nEmbedding matrix shape: {embeddings.shape}")

    # Save embeddings
    np.save(OUTPUT_EMB, embeddings)
    print(f"Embeddings saved → {OUTPUT_EMB}")

    # Save metadata
    meta = []
    for i, row in enumerate(records):
        meta.append({
            "index":               i,
            "id":                  row.get("id"),
            "title":               (row.get("title") or "").strip().replace("\n", " "),
            "doi":                 row.get("doi"),
            "primary_institution": row.get("primary_institution"),
            "region":              row.get("region"),
            "categories":          row.get("categories"),
        })

    with open(OUTPUT_META, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"Metadata saved  → {OUTPUT_META}")

    # verification
    loaded = np.load(OUTPUT_EMB)
    print(f"\nVerification — loaded shape: {loaded.shape}")
    print(f"First embedding (first 5 dims): {loaded[0][:5]}")
    print("\nDone!")


if __name__ == "__main__":
    main()
