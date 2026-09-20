"""
Task 4: Generation with Citations (Powered by Google Gemini)
"""

import os
import sys
from typing import List, Dict, Any, Tuple
from dotenv import load_dotenv

# Đảm bảo in tiếng Việt trên console Windows không bị lỗi cp1252
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

load_dotenv()


def format_context_with_citations(documents: List[Any]) -> Tuple[str, List[Dict[str, str]]]:
    """Tạo ngữ cảnh đánh số [1], [2],... và trích xuất danh sách metadata citations."""
    if not documents:
        return "", []

    context_blocks = []
    citations = []

    for idx, doc in enumerate(documents, start=1):
        if isinstance(doc, dict):
            content = doc.get("content") or doc.get("text", "")
            meta = doc.get("metadata", {})
        else:
            content = getattr(doc, "content", "")
            meta = getattr(doc, "metadata", {})

        title = meta.get("title") or meta.get("source") or f"Tài liệu #{idx}"
        url = meta.get("url") or meta.get("link") or "Không có liên kết"
        date = meta.get("published_date") or meta.get("date") or "N/A"

        context_blocks.append(
            f"--- [TÀI LIỆU {idx}] ---\n"
            f"Tiêu đề: {title}\n"
            f"Ngày xuất bản: {date}\n"
            f"Nội dung: {content}\n"
        )

        citations.append({"index": idx, "title": title, "url": url, "date": date})

    return "\n".join(context_blocks), citations


SYSTEM_PROMPT = """Bạn là trợ lý AI chuyên nghiệp về tin tức báo chí. Nhiệm vụ của bạn là trả lời câu hỏi của người dùng CHỈ dựa trên các tài liệu được cung cấp dưới đây.

Quy tắc bắt buộc:
1. Grounding: Chỉ sử dụng thông tin có trong 'Dữ liệu tham khảo'. Tuyệt đối không bịa đặt ngoài ngữ cảnh.
2. Citations: Cuối mỗi câu trả lời hoặc luận điểm, BẮT BUỘC ghi rõ số thứ tự tài liệu trích dẫn trong ngoặc vuông, ví dụ [1] hoặc [1][2].
3. Nếu không có thông tin trong tài liệu tham khảo, trả lời chính xác: 'Xin lỗi, tôi không tìm thấy thông tin phù hợp trong các bản tin được cung cấp.'
"""


def _call_gemini(prompt: str, api_key: str, model_name: str = "gemini-2.5-flash") -> str:
    """Hỗ trợ gọi Gemini linh hoạt bằng SDK google-genai mới hoặc google-generativeai cũ."""
    # Cách 1: Thử SDK google-genai mới
    try:
        from google import genai
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model=model_name,
            contents=prompt,
            config={"system_instruction": SYSTEM_PROMPT, "temperature": 0.2}
        )
        return response.text
    except ImportError:
        pass
    except Exception as e:
        # Nếu model gemini-2.5-flash chưa hỗ trợ ở SDK hiện tại, fallback thử gemini-1.5-flash
        try:
            from google import genai
            client = genai.Client(api_key=api_key)
            response = client.models.generate_content(
                model="gemini-1.5-flash",
                contents=prompt,
                config={"system_instruction": SYSTEM_PROMPT, "temperature": 0.2}
            )
            return response.text
        except Exception:
            pass

    # Cách 2: Thử google-generativeai SDK cũ
    try:
        import google.generativeai as genai_legacy
        genai_legacy.configure(api_key=api_key)
        model = genai_legacy.GenerativeModel(
            model_name="gemini-1.5-flash",
            system_instruction=SYSTEM_PROMPT
        )
        response = model.generate_content(prompt)
        return response.text
    except ImportError:
        return "[Lỗi: Chưa cài thư viện Gemini. Hãy chạy: pip install google-genai]"
    except Exception as e:
        return f"[Lỗi gọi Gemini API: {e}]"


def generate_answer(
    query: str,
    retrieved_docs: List[Any],
    api_key: str = None,
    model_name: str = "gemini-2.5-flash"
) -> Dict[str, Any]:
    """Sinh câu trả lời kèm citation sử dụng Gemini API."""
    context_str, citations = format_context_with_citations(retrieved_docs)

    if not context_str.strip():
        return {
            "answer": "Xin lỗi, tôi không tìm thấy thông tin phù hợp trong cơ sở dữ liệu tin tức được cung cấp.",
            "citations": [],
            "raw_context": ""
        }

    full_prompt = f"Dữ liệu tham khảo:\n{context_str}\n\nCâu hỏi: {query}\n\nCâu trả lời (hãy nhớ kèm trích dẫn [x]):"

    # Lấy API key từ biến truyền vào hoặc file .env
    gemini_key = (
        api_key
        or os.getenv("GEMINI_API_KEY")
        or os.getenv("gemini_api_key")
        or os.getenv("GOOGLE_API_KEY")
    )

    if not gemini_key:
        return {
            "answer": "[CẢNH BÁO: Chưa tìm thấy GEMINI_API_KEY trong file .env!]",
            "citations": citations,
            "raw_context": context_str
        }

    raw_answer = _call_gemini(full_prompt, api_key=gemini_key, model_name=model_name)

    # Thêm danh mục nguồn ở cuối bài
    citation_footer = "\n\n### Nguồn tham khảo:\n" + "\n".join(
        [f"- [{c['index']}] [{c['title']}]({c['url']}) ({c['date']})" for c in citations]
    )

    return {
        "answer": raw_answer + citation_footer,
        "citations": citations,
        "raw_context": context_str
    }


if __name__ == "__main__":
    print("\n--- TEST RUN TASK 4: GENERATION WITH CITATIONS ---")
    sample_docs = [
        {
            "content": "VinUni vừa tổ chức sự kiện Ngày hội AI 2026 với hơn 1000 người tham dự.",
            "metadata": {"title": "Ngày hội AI tại VinUni", "url": "https://vinuni.edu.vn/ai-day", "date": "2026-09-20"}
        },
        {
            "content": "Chương trình tập trung vào các giải pháp RAG và Agentic AI tiên tiến.",
            "metadata": {"title": "Xu hướng công nghệ 2026", "url": "https://techz.vn/ai-trend", "date": "2026-09-19"}
        }
    ]
    res = generate_answer("Sự kiện AI có những điểm gì nổi bật?", sample_docs)
    print(res["answer"])
