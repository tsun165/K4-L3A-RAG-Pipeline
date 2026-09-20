"""
Task 3 — Chuẩn hóa dữ liệu sang Markdown.

Hướng dẫn:
    1. Dùng MarkItDown để convert PDF/DOCX.
    2. Đọc JSON và giữ metadata ở đầu file Markdown.
    3. Giữ cấu trúc thư mục legal/ và news/.
    4. Không tạo file rỗng hoặc file trùng khi chạy lại.

Cài đặt:
    Dependency MarkItDown đã được khai báo trong pyproject.toml.
    
-> Hoặc dùng công cụ nào bạn quen khác Markitdown
"""

from pathlib import Path


LANDING_DIR = Path(__file__).parent.parent / "data" / "landing"
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "standardized"

# File PDF gốc từ chinhphu.vn là bản scan (không có text layer) nên
# MarkItDown trả về nội dung rỗng. Dùng bản toàn văn HTML tương đương
# từ nguồn công khai khác làm fallback để chuẩn hóa được nội dung thật.
FALLBACK_TEXT_SOURCES = {
    "nghi-dinh-105-2021-nd-cp-huong-dan-luat-phong-chong-ma-tuy": (
        "https://www.hethongphapluat.com/"
        "nghi-dinh-105-2021-nd-cp-huong-dan-luat-phong-chong-ma-tuy/noi-dung"
    ),
    "nghi-dinh-144-2021-nd-cp-xu-phat-vi-pham-hanh-chinh": (
        "https://vanban.vcci.com.vn/"
        "nghi-dinh-1442021nd-cp-quy-dinh-ve-xu-phat-vi-pham-hanh-chinh-trong"
        "-linh-vuc-an-ninh-trat-tu-an-toan-xa-hoi-phong-chong-te-nan-xa-hoi"
        "-phong-chay-chua-chay-cuu-nan-cuu-ho-phong-chong-bao-luc-gia-dinh"
    ),
}


def _trim_legal_markdown(markdown: str) -> str:
    """Cắt bỏ menu/footer của trang nguồn, giữ lại phần toàn văn nghị định."""
    start = markdown.find("Chương I")
    start = 0 if start == -1 else start

    end = len(markdown)
    for marker in ("Xem thêm", "⋮"):
        idx = markdown.find(marker, start)
        if idx != -1:
            end = min(end, idx)

    return markdown[start:end].strip()


async def _fetch_fallback_markdown(url: str) -> str:
    from crawl4ai import AsyncWebCrawler

    async with AsyncWebCrawler() as crawler:
        result = await crawler.arun(url=url)
        return _trim_legal_markdown(result.markdown)


def convert_legal_docs() -> None:
    """Convert PDF/DOCX vào standardized/legal."""
    import asyncio

    from markitdown import MarkItDown

    legal_dir = LANDING_DIR / "legal"
    output_dir = OUTPUT_DIR / "legal"
    output_dir.mkdir(parents=True, exist_ok=True)
    converter = MarkItDown()
    for path in sorted(legal_dir.iterdir()):
        if path.suffix.lower() not in {".pdf", ".doc", ".docx"}:
            continue

        content = converter.convert(str(path)).text_content.strip()
        source_note = ""
        if not content and path.stem in FALLBACK_TEXT_SOURCES:
            fallback_url = FALLBACK_TEXT_SOURCES[path.stem]
            content = asyncio.run(_fetch_fallback_markdown(fallback_url))
            source_note = (
                f"<!-- File gốc {path.name} là bản scan không có text layer. "
                f"Nội dung toàn văn được lấy từ: {fallback_url} -->\n\n"
            )

        if not content:
            print(f"Skipped (empty): {path.name}")
            continue
        (output_dir / f"{path.stem}.md").write_text(
            source_note + content, encoding="utf-8"
        )
        print(f"Converted: {path.name}")


def convert_news_articles() -> None:
    """Convert JSON vào standardized/news."""
    import json

    news_dir = LANDING_DIR / "news"
    output_dir = OUTPUT_DIR / "news"
    output_dir.mkdir(parents=True, exist_ok=True)
    for path in sorted(news_dir.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        content = data["content_markdown"].strip()
        if not content:
            print(f"Skipped (empty): {path.name}")
            continue
        header = (
            f"# {data['title']}\n\n"
            f"**Source:** {data['url']}\n\n"
            f"**Crawled:** {data['date_crawled']}\n\n---\n\n"
        )
        (output_dir / f"{path.stem}.md").write_text(
            header + content, encoding="utf-8"
        )
        print(f"Converted: {path.name}")


def convert_all() -> None:
    """Convert toàn bộ dữ liệu landing."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    convert_legal_docs()
    convert_news_articles()
    print(f"Saved Markdown to: {OUTPUT_DIR}")


if __name__ == "__main__":
    convert_all()
