"""
Task 5 — Semantic search.

Embed query bằng chính hàm của Task 4, query ChromaDB và đổi cosine distance
thành similarity. Output phải theo SearchResult, sort giảm dần và không quá top_k.

Score trả về là cosine similarity gốc (0..1) — Task 9 dùng chính score này để
quyết định fallback, không dùng RRF score.
"""

from .retrieval_utils import make_search_result
from .task4_chunking_indexing import embed_texts, get_collection


def _distance_to_score(distance: float) -> float:
    """Chroma với ``hnsw:space=cosine`` trả ``distance = 1 - cosine_similarity``.

    Cosine similarity có thể âm; kẹp về [0, 1] để threshold ở Task 9 đơn giản.
    """
    return max(0.0, min(1.0, 1.0 - float(distance)))


def _first_row(response: dict, key: str) -> list:
    """Lấy hàng kết quả đầu tiên (ta chỉ query 1 vector) và chịu được key thiếu."""
    rows = response.get(key) or []
    return list(rows[0]) if rows else []


def semantic_search(query: str, top_k: int = 10) -> list[dict]:
    """Trả về dense SearchResult theo score giảm dần."""
    if top_k <= 0 or not query or not query.strip():
        return []

    collection = get_collection()

    # Chroma cảnh báo/lỗi khi n_results lớn hơn số phần tử trong index.
    n_results = top_k
    count = getattr(collection, "count", None)
    if callable(count):
        available = int(count())
        if available <= 0:
            return []
        n_results = min(top_k, available)

    query_vector = embed_texts([query])[0]
    if hasattr(query_vector, "tolist"):
        query_vector = query_vector.tolist()

    response = collection.query(
        query_embeddings=[query_vector],
        n_results=n_results,
        include=["documents", "metadatas", "distances"],
    )

    ids = _first_row(response, "ids")
    documents = _first_row(response, "documents")
    metadatas = _first_row(response, "metadatas")
    distances = _first_row(response, "distances")

    results: list[dict] = []
    seen: set[str] = set()
    for item_id, content, metadata, distance in zip(ids, documents, metadatas, distances):
        if item_id in seen or not content:
            continue
        seen.add(item_id)
        results.append(
            make_search_result(
                item_id, content, _distance_to_score(distance), metadata, "dense"
            )
        )

    results.sort(key=lambda item: item["score"], reverse=True)
    return results[:top_k]


if __name__ == "__main__":
    for demo_query in (
        "Người sử dụng trái phép chất ma túy bị xử lý như thế nào?",
        "Ca sĩ Miu Lê bị bắt vì tội gì?",
    ):
        print(f"\n=== {demo_query}")
        for result in semantic_search(demo_query, top_k=3):
            title = result["metadata"].get("title", "")
            print(f"{result['score']:.3f}  {result['id']}  ({title})")
