"""
Task 3: Fusion & Fallback
- Thuật toán: Reciprocal Rank Fusion (RRF)
- Cơ chế: Fallback tự động khi 1 trong 2 retriever bị lỗi hoặc rỗng
- Chuẩn hóa: Tuân thủ contract đầu ra đồng nhất cho Task 4
"""

import logging
import sys
from typing import List, Dict, Any, Union
from dataclasses import dataclass, field

# Đảm bảo in tiếng Việt trên console Windows không bị lỗi cp1252
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("FusionFallback")


@dataclass
class StandardDocument:
    id: str
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    score: float = 0.0


def _extract_doc_info(doc: Union[dict, Any]) -> tuple:
    """Trích xuất dữ liệu linh hoạt, nhận cả dict lẫn class object từ Task 2."""
    if isinstance(doc, dict):
        doc_id = str(doc.get("id") or doc.get("doc_id") or doc.get("chunk_id") or hash(doc.get("content", "")))
        content = doc.get("content") or doc.get("text") or doc.get("page_content") or ""
        metadata = doc.get("metadata") or {}
        score = float(doc.get("score") or 0.0)
    else:
        doc_id = str(getattr(doc, "id", None) or getattr(doc, "doc_id", None) or hash(getattr(doc, "content", "")))
        content = getattr(doc, "content", None) or getattr(doc, "text", None) or getattr(doc, "page_content", "") or ""
        metadata = getattr(doc, "metadata", {}) or {}
        score = float(getattr(doc, "score", 0.0) or 0.0)
    return doc_id, content, metadata, score


def reciprocal_rank_fusion(
    dense_results: List[Any],
    sparse_results: List[Any],
    k: int = 60,
    top_k: int = 5
) -> List[StandardDocument]:
    """
    Kết hợp kết quả tìm kiếm từ Dense (Vector) và Sparse (BM25) bằng thuật toán RRF.
    Công thức: RRF_score(d) = sum( 1 / (k + rank_i(d)) )
    """
    rrf_scores: Dict[str, float] = {}
    doc_lookup: Dict[str, StandardDocument] = {}

    for rank, doc in enumerate(dense_results, start=1):
        doc_id, content, meta, _ = _extract_doc_info(doc)
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + (1.0 / (k + rank))
        if doc_id not in doc_lookup:
            doc_lookup[doc_id] = StandardDocument(id=doc_id, content=content, metadata=meta)

    for rank, doc in enumerate(sparse_results, start=1):
        doc_id, content, meta, _ = _extract_doc_info(doc)
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + (1.0 / (k + rank))
        if doc_id not in doc_lookup:
            doc_lookup[doc_id] = StandardDocument(id=doc_id, content=content, metadata=meta)

    # Sắp xếp lại theo điểm RRF giảm dần
    sorted_docs = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

    final_results = []
    for doc_id, score in sorted_docs[:top_k]:
        doc_obj = doc_lookup[doc_id]
        doc_obj.score = round(score, 5)
        final_results.append(doc_obj)

    return final_results


def hybrid_search_with_fallback(
    dense_results: List[Any] = None,
    sparse_results: List[Any] = None,
    k: int = 60,
    top_k: int = 5
) -> List[StandardDocument]:
    """
    Thực hiện RRF kèm Fallback thông minh:
    - Nếu cả hai có dữ liệu -> RRF fusion.
    - Nếu chỉ Dense có dữ liệu -> Fallback lấy Dense.
    - Nếu chỉ Sparse có dữ liệu -> Fallback lấy Sparse.
    - Nếu cả hai đều rỗng/lỗi -> Trả về danh sách rỗng an toàn.
    """
    has_dense = bool(dense_results and len(dense_results) > 0)
    has_sparse = bool(sparse_results and len(sparse_results) > 0)

    if has_dense and has_sparse:
        logger.info(f"Fusion: Kết hợp {len(dense_results)} dense và {len(sparse_results)} sparse docs.")
        return reciprocal_rank_fusion(dense_results, sparse_results, k=k, top_k=top_k)

    if has_dense and not has_sparse:
        logger.warning("Fallback: Sparse results bị rỗng/lỗi -> Sử dụng Dense results.")
        res = []
        for doc in dense_results[:top_k]:
            doc_id, content, meta, score = _extract_doc_info(doc)
            res.append(StandardDocument(id=doc_id, content=content, metadata=meta, score=score))
        return res

    if not has_dense and has_sparse:
        logger.warning("Fallback: Dense results bị rỗng/lỗi -> Sử dụng Sparse results.")
        res = []
        for doc in sparse_results[:top_k]:
            doc_id, content, meta, score = _extract_doc_info(doc)
            res.append(StandardDocument(id=doc_id, content=content, metadata=meta, score=score))
        return res

    logger.error("Fallback: Cả Dense và Sparse đều không trả về dữ liệu!")
    return []


if __name__ == "__main__":
    print("\n--- TEST RUN TASK 3: FUSION & FALLBACK ---")
    mock_dense = [
        {"id": "doc_1", "text": "Hội nghị AI quốc tế diễn ra tại Hà Nội.", "metadata": {"url": "https://news.vn/ai1", "title": "Hội nghị AI"}, "score": 0.92},
        {"id": "doc_2", "text": "Giá vàng hôm nay tăng mạnh.", "metadata": {"url": "https://news.vn/gold", "title": "Giá vàng"}, "score": 0.85},
    ]
    mock_sparse = [
        {"id": "doc_3", "text": "Các chuyên gia công nghệ bàn về AI.", "metadata": {"url": "https://news.vn/ai2", "title": "Bàn về AI"}, "score": 14.2},
        {"id": "doc_1", "text": "Hội nghị AI quốc tế diễn ra tại Hà Nội.", "metadata": {"url": "https://news.vn/ai1", "title": "Hội nghị AI"}, "score": 11.5},
    ]

    fused = hybrid_search_with_fallback(mock_dense, mock_sparse, top_k=3)
    for i, doc in enumerate(fused, 1):
        print(f"Top {i}: [{doc.id}] (Score: {doc.score}) - {doc.content} | Source: {doc.metadata.get('title')}")
