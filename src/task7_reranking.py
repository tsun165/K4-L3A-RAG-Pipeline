"""
Task 7 — Reciprocal Rank Fusion.

RRF gộp nhiều bảng xếp hạng mà không cộng trực tiếp cosine score với BM25
score. Công thức: RRF(d) = sum(1 / (k + rank)), rank bắt đầu từ 1.

Lưu ý: RRF score chỉ phản ánh thứ hạng, không dùng để quyết định fallback.

Quy tắc:
- Mỗi list được dedupe theo id trước khi tính rank (id trùng chỉ tính lần đầu).
- Content/metadata lấy từ lần xuất hiện đầu tiên của id (theo thứ tự list).
- Hoà điểm: ưu tiên phần tử xuất hiện trước (list đầu, rank cao hơn) để kết
  quả ổn định giữa các lần chạy.
- Không mutate input; mỗi kết quả là dict mới với ``retrieval_method="hybrid"``.
"""


def rerank_rrf(
    ranked_lists: list[list[dict]],
    top_k: int = 5,
    k: int = 60,
) -> list[dict]:
    """Fuse nhiều ranked lists và trả hybrid SearchResult."""
    if top_k <= 0:
        return []
    if k <= 0:
        raise ValueError("k phải là số dương")

    scores: dict[str, float] = {}
    items: dict[str, dict] = {}
    first_seen: dict[str, int] = {}

    for ranked_list in ranked_lists or []:
        seen_in_list: set[str] = set()
        rank = 0
        for item in ranked_list or []:
            item_id = item["id"]
            if item_id in seen_in_list:
                continue
            seen_in_list.add(item_id)
            rank += 1
            scores[item_id] = scores.get(item_id, 0.0) + 1.0 / (k + rank)
            if item_id not in items:
                items[item_id] = item
                first_seen[item_id] = len(first_seen)

    ranked_ids = sorted(scores, key=lambda item_id: (-scores[item_id], first_seen[item_id]))

    results: list[dict] = []
    for item_id in ranked_ids[:top_k]:
        source = items[item_id]
        results.append(
            {
                "id": source["id"],
                "content": source["content"],
                "score": scores[item_id],
                "metadata": dict(source.get("metadata") or {}),
                "retrieval_method": "hybrid",
            }
        )
    return results


if __name__ == "__main__":
    from .task5_semantic_search import semantic_search
    from .task6_lexical_search import lexical_search

    demo_query = "Người sử dụng trái phép chất ma túy bị xử phạt bao nhiêu tiền?"
    dense = semantic_search(demo_query, top_k=10)
    sparse = lexical_search(demo_query, top_k=10)
    print(f"dense={len(dense)} bm25={len(sparse)}")
    for result in rerank_rrf([dense, sparse], top_k=5):
        title = result["metadata"].get("title", "")
        print(f"{result['score']:.4f}  {result['id']}  ({title})")
