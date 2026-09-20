"""
Task 1 — Thu thập tài liệu chính sách/quy định.

Hướng dẫn:
    1. Chọn chủ đề của nhóm.
    2. Tìm tối thiểu 3 tài liệu PDF/DOCX từ nguồn công khai.
    3. Lưu file gốc vào data/landing/legal/.
    4. Đặt tên không dấu và thể hiện đúng nội dung.

Ví dụ tài liệu: học phí, học bổng, ký túc xá, quy trình đăng ký.
Nếu website chặn crawler, hãy chọn nguồn công khai khác; không vượt WAF.
"""

from pathlib import Path


DATA_DIR = Path(__file__).parent.parent / "data" / "landing" / "legal"


def setup_directory() -> None:
    """Tạo thư mục lưu tài liệu gốc."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Ready: {DATA_DIR}")


SOURCES = {
    "luat-73-2021-qh14-phong-chong-ma-tuy.pdf": (
        "https://datafiles.chinhphu.vn/cpp/files/vbpq/2022/01/73luat.pdf"
    ),
    "nghi-dinh-105-2021-nd-cp-huong-dan-luat-phong-chong-ma-tuy.pdf": (
        "https://datafiles.chinhphu.vn/cpp/files/vbpq/2021/12/105.signed_02.pdf"
    ),
    "nghi-dinh-144-2021-nd-cp-xu-phat-vi-pham-hanh-chinh.pdf": (
        "https://datafiles.chinhphu.vn/cpp/files/vbpq/2022/01/144.signed.pdf"
    ),
}


def download_documents() -> None:
    """Tải ít nhất 3 PDF/DOCX từ nguồn công khai."""
    import requests

    headers = {"User-Agent": "Mozilla/5.0 (compatible; RAGLabBot/1.0)"}
    for filename, url in SOURCES.items():
        destination = DATA_DIR / filename
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        destination.write_bytes(response.content)
        print(f"Downloaded: {destination}")


if __name__ == "__main__":
    setup_directory()
    download_documents()
