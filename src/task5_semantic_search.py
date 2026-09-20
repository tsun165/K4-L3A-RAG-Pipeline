"""
Task 5 — Semantic search.

Embed query bằng chính hàm của Task 4, query ChromaDB và đổi cosine distance
thành similarity. Output phải theo SearchResult, sort giảm dần và không quá top_k.
"""

from .task4_chunking_indexing import embed_texts, get_collection


def semantic_search(query: str, top_k: int = 10) -> list[dict]:
    """Trả về dense SearchResult theo score giảm dần."""
    query_vectors = embed_texts([query])
    if not query_vectors:
        return []
    query_vector = query_vectors[0]

    collection = get_collection()
    response = collection.query(
        query_embeddings=[query_vector],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    results = []
    if response and response.get("ids") and len(response["ids"]) > 0:
        for item_id, content, metadata, distance in zip(
            response["ids"][0],
            response["documents"][0],
            response["metadatas"][0],
            response["distances"][0],
        ):
            score = max(0.0, 1.0 - float(distance))
            results.append({
                "id": item_id,
                "content": content,
                "score": score,
                "metadata": metadata,
                "retrieval_method": "dense",
            })

    results.sort(key=lambda item: item["score"], reverse=True)
    return results[:top_k]


if __name__ == "__main__":
    for result in semantic_search("test query", top_k=3):
        print(result)
