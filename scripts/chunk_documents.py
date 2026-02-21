import json
from tqdm import tqdm


def chunk_text(text, chunk_size=300, overlap=50):
    words = text.split()
    chunks = []

    for i in range(0, len(words), chunk_size - overlap):
        chunk = words[i:i + chunk_size]
        if chunk:
            chunks.append(" ".join(chunk))

    return chunks


def create_chunks(input_path, output_path):

    with open(input_path, "r", encoding="utf-8") as f:
        papers = json.load(f)

    all_chunks = []

    for paper in tqdm(papers):

        text = paper.get("abstract", "") + " " + paper.get("full_text", "")

        chunks = chunk_text(text)

        for idx, chunk in enumerate(chunks):
            all_chunks.append({
                "paper_id": paper["paper_id"],
                "title": paper.get("title", ""),
                "authors": paper.get("authors", ""),
                "year": paper.get("year", 0),
                "venue": paper.get("venue", ""),
                "citation_count": paper.get("citation_count", 0),
                "chunk_index": idx,
                "chunk_text": chunk
            })

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, indent=2)

    print(f"Created {len(all_chunks)} chunks.")


if __name__ == "__main__":
    create_chunks(
        "data/enriched_data.json",
        "data/processed_chunks.json"
    )
