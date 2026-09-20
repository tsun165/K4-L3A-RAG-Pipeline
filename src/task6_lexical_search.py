"""
Task 6 — Lexical search bằng BM25.

Dùng cùng corpus chunks với Task 5. BM25 phù hợp với từ khóa chính xác, mã tài
liệu và tên riêng. Output phải theo SearchResult và sort score giảm dần.

Thiết kế:
- Corpus: nạp từ ChromaDB (``load_corpus``) để id/content/metadata trùng khớp
  100% với dense search — RRF ở Task 7 fuse theo id nên điều này bắt buộc.
  Chạy ``python -m src.task4_chunking_indexing`` trước.
- Tokenizer tiếng Việt: tách theo âm tiết (unicode ``\\w+``) rồi ghép thêm bigram
  âm tiết liền kề, ví dụ "ma túy" -> ["ma", "túy", "ma_túy"]. Bigram giúp BM25
  khớp từ ghép mà không cần thêm dependency tách từ (pyvi/underthesea).
- IDF: dùng công thức Lucene ``log(1 + (N - n + 0.5) / (n + 0.5))`` thay cho
  IDF gốc của ``rank_bm25`` (bằng 0 hoặc âm khi corpus nhỏ, ví dụ 2 doc).
- Index được cache theo corpus đang dùng; đổi ``CORPUS`` là tự rebuild.
"""

import math
import re
import unicodedata

import numpy as np
from rank_bm25 import BM25Okapi

from .retrieval_utils import make_search_result, normalize_metadata


CORPUS: list[dict] = []

USE_BIGRAMS = True
BM25_K1 = 1.5
BM25_B = 0.75

_TOKEN_PATTERN = re.compile(r"\w+", re.UNICODE)
_INDEX_CACHE: dict = {"fingerprint": None, "index": None}


class LuceneBM25(BM25Okapi):
    """BM25Okapi với IDF luôn dương (công thức của Lucene/Elasticsearch)."""

    def _calc_idf(self, nd: dict) -> None:
        self.idf = {
            word: math.log(1.0 + (self.corpus_size - freq + 0.5) / (freq + 0.5))
            for word, freq in nd.items()
        }
        self.average_idf = (
            sum(self.idf.values()) / len(self.idf) if self.idf else 0.0
        )


def tokenize(text: str) -> list[str]:
    """Tách âm tiết tiếng Việt (lowercase, NFC) và thêm bigram liền kề."""
    text = unicodedata.normalize("NFC", text or "").lower()
    syllables = [token for token in _TOKEN_PATTERN.findall(text) if token != "_"]
    if not USE_BIGRAMS or len(syllables) < 2:
        return syllables
    bigrams = [f"{left}_{right}" for left, right in zip(syllables, syllables[1:])]
    return syllables + bigrams


def load_corpus() -> list[dict]:
    """Nạp toàn bộ chunks từ ChromaDB vào ``CORPUS`` (giữ nguyên object list)."""
    from .task4_chunking_indexing import get_collection

    response = get_collection().get(include=["documents", "metadatas"])
    items: list[dict] = []
    for item_id, content, metadata in zip(
        response.get("ids") or [],
        response.get("documents") or [],
        response.get("metadatas") or [],
    ):
        if not content:
            continue
        items.append(
            {"id": item_id, "content": content, "metadata": normalize_metadata(metadata)}
        )
    items.sort(
        key=lambda item: (
            item["metadata"].get("source", ""),
            item["metadata"].get("chunk_index", 0),
        )
    )
    if not items:
        raise RuntimeError(
            "ChromaDB collection rỗng. Chạy `python -m src.task4_chunking_indexing` "
            "trước khi dùng lexical_search."
        )
    CORPUS[:] = items
    invalidate_index()
    return CORPUS


def build_bm25_index(corpus: list[dict]):
    """Tạo BM25 index từ cùng corpus chunks của Task 4."""
    if not corpus:
        raise ValueError("corpus rỗng, không thể tạo BM25 index")
    tokenized = [tokenize(item["content"]) for item in corpus]
    return LuceneBM25(tokenized, k1=BM25_K1, b=BM25_B)


def invalidate_index() -> None:
    """Buộc rebuild BM25 index ở lần search kế tiếp."""
    _INDEX_CACHE["fingerprint"] = None
    _INDEX_CACHE["index"] = None


def _fingerprint(corpus: list[dict]) -> tuple:
    first = corpus[0]["id"] if corpus else None
    last = corpus[-1]["id"] if corpus else None
    return (id(corpus), len(corpus), first, last)


def _get_index(corpus: list[dict]):
    fingerprint = _fingerprint(corpus)
    if _INDEX_CACHE["index"] is None or _INDEX_CACHE["fingerprint"] != fingerprint:
        _INDEX_CACHE["index"] = build_bm25_index(corpus)
        _INDEX_CACHE["fingerprint"] = fingerprint
    return _INDEX_CACHE["index"]


def lexical_search(query: str, top_k: int = 10) -> list[dict]:
    """Trả về BM25 SearchResult theo score giảm dần (chỉ giữ score > 0)."""
    if top_k <= 0:
        return []
    query_tokens = tokenize(query)
    if not query_tokens:
        return []

    corpus = CORPUS if CORPUS else load_corpus()
    if not corpus:
        return []

    scores = np.asarray(_get_index(corpus).get_scores(query_tokens), dtype=float)
    order = np.argsort(-scores, kind="stable")  # tie -> giữ thứ tự corpus

    results: list[dict] = []
    seen: set[str] = set()
    for index in order:
        score = float(scores[index])
        if score <= 0.0 or len(results) >= top_k:
            break
        item = corpus[int(index)]
        if item["id"] in seen:
            continue
        seen.add(item["id"])
        results.append(
            make_search_result(item["id"], item["content"], score, item["metadata"], "bm25")
        )
    return results


if __name__ == "__main__":
    corpus = load_corpus()
    print(f"BM25 corpus: {len(corpus)} chunks")
    for demo_query in (
        "Nghị định 144/2021/NĐ-CP xử phạt hành vi sử dụng trái phép chất ma túy",
        "Tăng Nhật Tuệ bị bắt",
    ):
        print(f"\n=== {demo_query}")
        for result in lexical_search(demo_query, top_k=3):
            title = result["metadata"].get("title", "")
            print(f"{result['score']:.3f}  {result['id']}  ({title})")
