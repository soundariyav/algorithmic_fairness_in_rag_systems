import json
from tqdm import tqdm
from chroma_config import collection


def load_chunks(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def index_chunks(data, batch_size=100):

    ids, docs, metas = [], [], []

    for doc in tqdm(data):

        chunk_id = f"{doc['paper_id']}_{doc['chunk_index']}"

        ids.append(chunk_id)
        docs.append(doc["chunk_text"])

        metas.append({
            "paper_id": doc["paper_id"],
            "title": doc["title"],
            "authors": doc["authors"],
            "year": doc["year"],
            "venue": doc["venue"],
            "citation_count": doc["citation_count"],
            "chunk_index": doc["chunk_index"]
        })

        if len(ids) >= batch_size:
            collection.add(ids=ids, documents=docs, metadatas=metas)
            ids, docs, metas = [], [], []

    if ids:
        collection.add(ids=ids, documents=docs, metadatas=metas)

    print("✅ All chunks indexed successfully.")


if __name__ == "__main__":
    data = load_chunks("data/processed_chunks.json")
    index_chunks(data)
