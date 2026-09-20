"""
Helper dùng chung cho các module retrieval (Task 5–7).

Mục đích: mọi SearchResult đi ra khỏi dense/BM25/RRF đều cùng một schema
(docs/MODULE_CONTRACTS.md) dù dữ liệu được đọc từ ChromaDB hay từ corpus
trong bộ nhớ.
"""

from .contracts import RetrievalMethod


def normalize_metadata(metadata: dict | None) -> dict:
    """Đưa metadata đọc từ vector store về đúng ``ChunkMetadata``.

    ChromaDB không lưu được giá trị ``None`` nên Task 4 có thể lưu ``url=""``
    hoặc bỏ hẳn key. Khi đọc lại ta khôi phục ``url=None`` và ép
    ``chunk_index`` về ``int`` để validator của contract chấp nhận.
    """
    normalized = dict(metadata or {})
    url = normalized.get("url")
    normalized["url"] = url if isinstance(url, str) and url.strip() else None
    if "chunk_index" in normalized:
        try:
            normalized["chunk_index"] = int(normalized["chunk_index"])
        except (TypeError, ValueError):
            pass
    return normalized


def make_search_result(
    item_id: str,
    content: str,
    score: float,
    metadata: dict | None,
    retrieval_method: RetrievalMethod,
) -> dict:
    """Tạo một SearchResult mới (không chia sẻ dict metadata với input)."""
    return {
        "id": str(item_id),
        "content": content,
        "score": float(score),
        "metadata": normalize_metadata(metadata),
        "retrieval_method": retrieval_method,
    }
