"""Unit test bổ sung cho Task 5–7 (retrieval). Không gọi network hay API thật."""

import pytest

from src.contracts import validate_search_results


def metadata(source: str = "luat-73.md", chunk_index: int = 0) -> dict:
    return {
        "source": source,
        "title": "Luật Phòng, chống ma túy",
        "doc_type": "legal",
        "url": None,
        "chunk_index": chunk_index,
    }


def chunk(item_id: str, content: str, chunk_index: int = 0) -> dict:
    return {"id": item_id, "content": content, "metadata": metadata(chunk_index=chunk_index)}


def result(item_id: str, score: float, method: str = "dense") -> dict:
    return {
        "id": item_id,
        "content": f"content of {item_id}",
        "score": score,
        "metadata": metadata(chunk_index=int(item_id.rsplit("-", 1)[-1])),
        "retrieval_method": method,
    }


# --- Task 6: tokenizer ------------------------------------------------------


def test_tokenize_lowercases_strips_punctuation_and_adds_bigrams():
    from src.task6_lexical_search import tokenize

    tokens = tokenize("Luật Phòng, chống MA TÚY 73/2021/QH14")

    assert "luật" in tokens and "phòng" in tokens
    assert "ma" in tokens and "túy" in tokens and "ma_túy" in tokens
    assert "73" in tokens and "qh14" in tokens and "73_2021" in tokens
    assert all("," not in token and "/" not in token for token in tokens)


def test_tokenize_handles_empty_and_single_syllable():
    from src.task6_lexical_search import tokenize

    assert tokenize("") == []
    assert tokenize("   ") == []
    assert tokenize("Hà") == ["hà"]


# --- Task 6: lexical search --------------------------------------------------


def test_lexical_search_prefers_phrase_match_via_bigrams(monkeypatch):
    import src.task6_lexical_search as lexical

    corpus = [
        chunk("chunk-0", "ma quỷ say túy lúy trong đêm", 0),
        chunk("chunk-1", "tổ chức sử dụng trái phép chất ma túy", 1),
        chunk("chunk-2", "thư viện mở cửa từ 8 giờ sáng", 2),
    ]
    monkeypatch.setattr(lexical, "CORPUS", corpus)

    output = lexical.lexical_search("ma túy", top_k=5)

    validate_search_results(output, top_k=5, expected_method="bm25")
    assert output[0]["id"] == "chunk-1"
    assert [item["id"] for item in output] == ["chunk-1", "chunk-0"]


def test_lexical_search_returns_empty_for_blank_query_or_zero_top_k(monkeypatch):
    import src.task6_lexical_search as lexical

    monkeypatch.setattr(lexical, "CORPUS", [chunk("chunk-0", "ma túy")])
    assert lexical.lexical_search("", top_k=5) == []
    assert lexical.lexical_search("!!! ...", top_k=5) == []
    assert lexical.lexical_search("ma túy", top_k=0) == []


def test_lexical_search_rebuilds_index_when_corpus_changes(monkeypatch):
    import src.task6_lexical_search as lexical

    monkeypatch.setattr(lexical, "CORPUS", [chunk("a-0", "học phí đại học", 0)])
    assert [item["id"] for item in lexical.lexical_search("học phí")] == ["a-0"]

    monkeypatch.setattr(lexical, "CORPUS", [chunk("b-0", "ký túc xá sinh viên", 0)])
    assert lexical.lexical_search("học phí") == []
    assert [item["id"] for item in lexical.lexical_search("ký túc xá")] == ["b-0"]


def test_lexical_search_does_not_share_metadata_with_corpus(monkeypatch):
    import src.task6_lexical_search as lexical

    corpus = [chunk("chunk-0", "ma túy", 0)]
    monkeypatch.setattr(lexical, "CORPUS", corpus)
    output = lexical.lexical_search("ma túy", top_k=1)

    output[0]["metadata"]["title"] = "changed"
    assert corpus[0]["metadata"]["title"] == "Luật Phòng, chống ma túy"


# --- Task 5: semantic search -------------------------------------------------


def test_semantic_search_clamps_score_and_normalizes_metadata(monkeypatch):
    import src.task5_semantic_search as semantic

    class FakeCollection:
        def count(self):
            return 2

        def query(self, **kwargs):
            assert kwargs["n_results"] == 2  # min(top_k, count)
            return {
                "ids": [["chunk-0", "chunk-1"]],
                "documents": [["Điều 1. Phạm vi điều chỉnh", "Điều 2. Giải thích từ ngữ"]],
                "metadatas": [
                    [
                        {"source": "a.md", "title": "A", "doc_type": "legal", "url": "", "chunk_index": 0},
                        {"source": "a.md", "title": "A", "doc_type": "legal", "chunk_index": "1"},
                    ]
                ],
                "distances": [[0.25, 1.4]],
            }

    monkeypatch.setattr(semantic, "embed_texts", lambda texts: [[0.5, 0.5]])
    monkeypatch.setattr(semantic, "get_collection", lambda: FakeCollection())

    output = semantic.semantic_search("phạm vi", top_k=5)

    validate_search_results(output, top_k=5, expected_method="dense")
    assert output[0]["score"] == pytest.approx(0.75)
    assert output[1]["score"] == 0.0  # 1 - 1.4 < 0 -> kẹp về 0
    assert output[0]["metadata"]["url"] is None
    assert output[1]["metadata"]["url"] is None
    assert output[1]["metadata"]["chunk_index"] == 1


def test_semantic_search_returns_empty_for_empty_collection_or_query(monkeypatch):
    import src.task5_semantic_search as semantic

    class EmptyCollection:
        def count(self):
            return 0

        def query(self, **kwargs):
            raise AssertionError("must not query an empty collection")

    monkeypatch.setattr(semantic, "embed_texts", lambda texts: [[0.5, 0.5]])
    monkeypatch.setattr(semantic, "get_collection", lambda: EmptyCollection())

    assert semantic.semantic_search("ma túy", top_k=3) == []
    assert semantic.semantic_search("   ", top_k=3) == []


# --- Task 7: RRF -------------------------------------------------------------


def test_rrf_handles_empty_lists_and_duplicates_within_a_list():
    from src.task7_reranking import rerank_rrf

    assert rerank_rrf([], top_k=5) == []
    assert rerank_rrf([[], []], top_k=5) == []

    dense = [result("chunk-0", 0.9), result("chunk-0", 0.8), result("chunk-1", 0.7)]
    fused = rerank_rrf([dense, []], top_k=5)

    assert [item["id"] for item in fused] == ["chunk-0", "chunk-1"]
    assert fused[0]["score"] == pytest.approx(1 / 61)
    assert fused[1]["score"] == pytest.approx(1 / 62)  # duplicate không chiếm rank


def test_rrf_tie_break_is_deterministic_and_respects_top_k():
    from src.task7_reranking import rerank_rrf

    dense = [result("chunk-0", 0.9), result("chunk-1", 0.8)]
    bm25 = [result("chunk-2", 9.0, "bm25"), result("chunk-3", 8.0, "bm25")]

    fused = rerank_rrf([dense, bm25], top_k=3)

    validate_search_results(fused, top_k=3, expected_method="hybrid")
    assert [item["id"] for item in fused] == ["chunk-0", "chunk-2", "chunk-1"]


def test_rrf_does_not_mutate_inputs():
    from src.task7_reranking import rerank_rrf

    dense = [result("chunk-0", 0.9)]
    bm25 = [result("chunk-0", 9.0, "bm25")]
    fused = rerank_rrf([dense, bm25], top_k=1)

    fused[0]["metadata"]["title"] = "changed"
    assert dense[0]["retrieval_method"] == "dense"
    assert dense[0]["score"] == 0.9
    assert dense[0]["metadata"]["title"] == "Luật Phòng, chống ma túy"
