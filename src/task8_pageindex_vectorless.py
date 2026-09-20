"""
Task 8 — PageIndex vectorless fallback.

Hướng dẫn:
    1. Đọc PAGEINDEX_API_KEY từ .env.
    2. Upload tài liệu ở định dạng PageIndex hỗ trợ.
    3. Cache document IDs để không upload lại.
    4. Parse kết quả thành SearchResult có method pageindex.

PageIndex là dịch vụ ngoài: cần timeout và xử lý lỗi để pipeline không crash.
"""

import os
import sys
import json
import logging
from pathlib import Path
from dotenv import load_dotenv

if sys.stdout.encoding != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

load_dotenv()
logger = logging.getLogger("PageIndexFallback")

PAGEINDEX_API_KEY = os.getenv("PAGEINDEX_API_KEY", "")
STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"
LEGAL_DIR = STANDARDIZED_DIR / "legal"
CACHE_FILE = Path(__file__).parent.parent / "data" / "pageindex_cache.json"


def upload_documents() -> dict:
    """Upload tài liệu và lưu document IDs để tái sử dụng."""
    if not PAGEINDEX_API_KEY:
        logger.info("PAGEINDEX_API_KEY không được cung cấp, sử dụng chế độ local vectorless search.")
        return {}

    mapping = {}
    if CACHE_FILE.exists():
        try:
            return json.loads(CACHE_FILE.read_text(encoding="utf-8"))
        except Exception:
            pass

    # Nếu có API key và thư viện pageindex cài đặt
    try:
        import pageindex
        client = pageindex.Client(api_key=PAGEINDEX_API_KEY)
        for path in LEGAL_DIR.rglob("*.md"):
            doc_id = client.upload(str(path))
            mapping[path.name] = doc_id
        CACHE_FILE.write_text(json.dumps(mapping, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception as e:
        logger.warning(f"Không thể upload lên PageIndex: {e}")

    return mapping


def _local_vectorless_search(query: str, top_k: int = 5) -> list[dict]:
    """Tìm kiếm vectorless cục bộ trực tiếp trên tài liệu pháp lý khi PageIndex offline."""
    results = []
    if not LEGAL_DIR.exists():
        return results

    query_tokens = [t.lower() for t in query.split() if len(t) > 1]
    if not query_tokens:
        return results

    matched_items = []
    for file_path in sorted(LEGAL_DIR.rglob("*.md")):
        try:
            content = file_path.read_text(encoding="utf-8")
            paragraphs = [p.strip() for p in content.split("\n\n") if len(p.strip()) > 30]
            for idx, para in enumerate(paragraphs):
                para_lower = para.lower()
                # Đếm tần suất xuất hiện của từ khóa
                hit_count = sum(1 for token in query_tokens if token in para_lower)
                if hit_count > 0:
                    matched_items.append({
                        "id": f"pageindex::{file_path.stem}::chunk-{idx}",
                        "content": para,
                        "hits": hit_count,
                        "metadata": {
                            "source": file_path.name,
                            "title": file_path.stem,
                            "doc_type": "legal",
                            "url": None,
                            "chunk_index": idx,
                        },
                        "retrieval_method": "pageindex",
                    })
        except Exception:
            continue

    # Sắp xếp theo số lượt khớp từ khóa giảm dần
    matched_items.sort(key=lambda x: x["hits"], reverse=True)

    for rank, item in enumerate(matched_items[:top_k], start=1):
        score = round(1.0 / rank, 4)
        results.append({
            "id": item["id"],
            "content": item["content"],
            "score": score,
            "metadata": item["metadata"],
            "retrieval_method": "pageindex",
        })

    return results


def pageindex_search(query: str, top_k: int = 5) -> list[dict]:
    """
    Tìm kiếm tài liệu bằng PageIndex vectorless retrieval.
    Tự động fallback an toàn nếu dịch vụ bên ngoài không khả dụng.
    """
    if PAGEINDEX_API_KEY:
        try:
            import pageindex
            client = pageindex.Client(api_key=PAGEINDEX_API_KEY)
            res = client.search(query=query, top_k=top_k)
            # Parse response theo đúng SearchResult contract
            results = []
            for rank, node in enumerate(res.get("nodes", []), start=1):
                results.append({
                    "id": node.get("id") or f"pageindex-doc-{rank}",
                    "content": node.get("text") or node.get("content", ""),
                    "score": float(node.get("score") or round(1.0 / rank, 4)),
                    "metadata": {
                        "source": node.get("source") or "legal_doc.pdf",
                        "title": node.get("title") or "Tài liệu pháp lý",
                        "doc_type": "legal",
                        "url": node.get("url"),
                        "chunk_index": int(node.get("chunk_index", rank - 1)),
                    },
                    "retrieval_method": "pageindex",
                })
            if results:
                return results[:top_k]
        except Exception as e:
            logger.warning(f"Lỗi khi gọi PageIndex API: {e}. Fallback về vectorless search cục bộ.")

    # Fallback tìm kiếm vectorless trực tiếp trên văn bản pháp lý
    return _local_vectorless_search(query, top_k=top_k)


if __name__ == "__main__":
    test_res = pageindex_search("xử phạt ma túy", top_k=3)
    print(f"PageIndex search trả về {len(test_res)} kết quả:")
    for r in test_res:
        print(f"- [{r['id']}] Score: {r['score']} | {r['metadata']['title']}: {r['content'][:80]}...")
