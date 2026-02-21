from chroma_config import collection


def search(query, k=5):

    results = collection.query(
        query_texts=[query],
        n_results=k
    )

    docs = results["documents"][0]
    metas = results["metadatas"][0]
    distances = results["distances"][0]

    for i in range(len(docs)):
        print("=" * 70)
        print("Title:", metas[i]["title"])
        print("Year:", metas[i]["year"])
        print("Venue:", metas[i]["venue"])
        print("Score:", 1 - distances[i])
        print("-" * 70)
        print(docs[i][:400])


if __name__ == "__main__":

    while True:
        q = input("Query (or 'exit'): ")
        if q.lower() == "exit":
            break
        search(q)
