"""
Task 6 — Lexical search bằng BM25.

Dùng cùng corpus chunks với Task 5. BM25 phù hợp với từ khóa chính xác, mã tài
liệu và tên riêng. Output phải theo SearchResult và sort score giảm dần.
"""

import math
from typing import Any
from rank_bm25 import BM25Okapi


class LuceneBM25Okapi(BM25Okapi):
    """
    BM25Okapi cải tiến với công thức IDF của Apache Lucene:
    IDF = ln(1 + (N - n + 0.5) / (n + 0.5))
    Đảm bảo IDF luôn dương cho mọi từ khóa xuất hiện trong corpus,
    tránh lỗi điểm bằng 0 khi corpus nhỏ (từ xuất hiện trong >= 50% văn bản).
    """
    def _calc_idf(self, nd):
        for word, freq in nd.items():
            self.idf[word] = math.log(1.0 + (self.corpus_size - freq + 0.5) / (freq + 0.5))


CORPUS: list[dict] = []
_BM25_INDEX = None
_INDEXED_CORPUS_LEN = 0


def _ensure_corpus():
    """Tự động nạp corpus từ standardized data nếu CORPUS hiện tại đang rỗng."""
    global CORPUS
    if not CORPUS:
        try:
            from .task4_chunking_indexing import load_documents, chunk_documents
            docs = load_documents()
            CORPUS = chunk_documents(docs)
        except Exception:
            pass


def build_bm25_index(corpus: list[dict]):
    """Tạo BM25 index từ cùng corpus chunks của Task 4."""
    tokenized = [item["content"].lower().split() for item in corpus]
    return LuceneBM25Okapi(tokenized)


def lexical_search(query: str, top_k: int = 10) -> list[dict]:
    """Trả về BM25 SearchResult theo score giảm dần."""
    global CORPUS, _BM25_INDEX, _INDEXED_CORPUS_LEN

    _ensure_corpus()
    if not CORPUS:
        return []

    # Cập nhật index nếu corpus thay đổi hoặc chưa khởi tạo
    if _BM25_INDEX is None or len(CORPUS) != _INDEXED_CORPUS_LEN:
        _BM25_INDEX = build_bm25_index(CORPUS)
        _INDEXED_CORPUS_LEN = len(CORPUS)

    query_tokens = query.lower().split()
    if not query_tokens:
        return []

    scores = _BM25_INDEX.get_scores(query_tokens)

    # Sắp xếp các tài liệu theo điểm BM25 giảm dần
    ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)

    results = []
    for idx in ranked_indices:
        if scores[idx] <= 0:
            continue
        item = CORPUS[idx]
        results.append({
            "id": item["id"],
            "content": item["content"],
            "score": float(scores[idx]),
            "metadata": item["metadata"],
            "retrieval_method": "bm25",
        })
        if len(results) >= top_k:
            break

    return results


if __name__ == "__main__":
    for result in lexical_search("test query", top_k=3):
        print(result)
