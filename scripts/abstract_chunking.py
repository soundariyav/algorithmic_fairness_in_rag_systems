import pandas as pd
import json
import re

# Config
INPUT_FILE = "cs_papers_100k_with_institutions.json"
OUTPUT_FILE = "abstract_chunks.json"

CHUNK_SIZE = 400
CHUNK_OVERLAP = 50


# -----------------------------
# Load dataset
# -----------------------------
df = pd.read_json(INPUT_FILE)
print("Total papers loaded:", len(df))


# -----------------------------
# Clean abstract text
# -----------------------------
def clean_text(text):
    if pd.isna(text):
        return ""
    text = re.sub(r"\s+", " ", str(text))
    return text.strip()


# -----------------------------
# Chunking function
# -----------------------------
def chunk_text(text, chunk_size=400, overlap=50):
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start += chunk_size - overlap

    return chunks


# -----------------------------
# Create chunks with metadata
# -----------------------------
output_data = []

for idx, row in df.iterrows():

    abstract = clean_text(row.get("abstract", ""))

    # skip very short abstracts
    if len(abstract) < 50:
        continue

    chunks = chunk_text(abstract, CHUNK_SIZE, CHUNK_OVERLAP)

    for i, chunk in enumerate(chunks):
        record = {
            "id": f"{idx}_chunk_{i}",
            "text": chunk,
            "metadata": {
                "doi": row.get("doi"),
                "primary_institution": row.get("primary_institution"),
                "region": row.get("region"),
                "chunk_index": i
            }
        }

        output_data.append(record)


print("Total chunks created:", len(output_data))


# -----------------------------
# Save JSON file
# -----------------------------
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(output_data, f, indent=2)

print("Output saved to:", OUTPUT_FILE)
