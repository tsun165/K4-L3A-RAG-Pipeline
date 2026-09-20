import os
import time
import streamlit as st
from dotenv import load_dotenv

from src.task10_generation import (
    SYSTEM_PROMPT,
    format_context,
    reorder_for_llm,
    call_llm,
)
from src.task9_retrieval_pipeline import retrieve
from src.task5_semantic_search import semantic_search
from src.task6_lexical_search import lexical_search
from src.task8_pageindex_vectorless import pageindex_search

load_dotenv()

# Cấu hình trang (Wide layout)
st.set_page_config(
    page_title="Hệ thống RAG Pháp luật & Phòng chống ma túy",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS giao diện (tương thích cả Dark Mode và Light Mode)
st.markdown(
    """
    <style>
    .main-title {
        font-size: 1.6rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
        color: var(--text-color, #1E3A8A);
    }
    .sub-title {
        font-size: 0.88rem;
        color: #94A3B8;
        margin-bottom: 0.8rem;
    }
    .citation-header {
        padding: 10px 14px;
        border-radius: 8px;
        margin-bottom: 12px;
        background-color: rgba(148, 163, 184, 0.1);
        border: 1px solid rgba(148, 163, 184, 0.2);
    }
    .meta-badge {
        display: inline-block;
        font-size: 0.75rem;
        font-weight: 600;
        padding: 2px 8px;
        border-radius: 4px;
        margin-right: 6px;
    }
    .badge-legal {
        background-color: rgba(59, 130, 246, 0.2);
        color: #60A5FA;
        border: 1px solid rgba(59, 130, 246, 0.4);
    }
    .badge-news {
        background-color: rgba(16, 185, 129, 0.2);
        color: #34D399;
        border: 1px solid rgba(16, 185, 129, 0.4);
    }
    .badge-method {
        background-color: rgba(245, 158, 11, 0.2);
        color: #FBBF24;
        border: 1px solid rgba(245, 158, 11, 0.4);
    }
    .score-badge {
        font-size: 0.82rem;
        color: #10B981;
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def mock_generate_with_citation(query: str, top_k: int = 5) -> dict:
    """Hàm giả lập GenerationResult theo đúng Module Contract."""
    q = query.lower()

    # Trường hợp 1: Query ngoài domain -> Safe refusal
    if any(k in q for k in ["thời tiết", "nấu ăn", "bóng đá", "tổng thống", "du lịch"]):
        return {
            "answer": "Tôi không thể xác minh thông tin này từ nguồn dữ liệu hiện có trong hệ thống pháp luật và tin tức về phòng, chống ma túy.",
            "sources": [],
            "retrieval_source": "none",
        }

    # Trường hợp 2: Vụ việc ca sĩ Miu Lê
    if "miu lê" in q or "cát bà" in q or "tùng thu" in q:
        sources = [
            {
                "id": "article_01-chunk-0",
                "content": "Ngày 16/5, Cơ quan CSĐT Công an thành phố Hải Phòng đã ra quyết định khởi tố, lệnh bắt tạm giam đối với Lê Ánh Nhật (ca sĩ Miu Lê, 35 tuổi, trú TP HCM) cùng Vũ Khương An về hành vi 'Tổ chức sử dụng trái phép chất ma túy'.",
                "score": 0.94,
                "metadata": {
                    "source": "article_01.md",
                    "title": "Ca sĩ Miu Lê bị bắt với cáo buộc tổ chức sử dụng ma túy",
                    "doc_type": "news",
                    "url": "https://vnexpress.net/ca-si-miu-le-bi-bat-voi-cao-buoc-to-chuc-su-dung-ma-tuy-5074769.html",
                    "chunk_index": 0,
                },
                "retrieval_method": "hybrid",
            },
            {
                "id": "article_01-chunk-1",
                "content": "Kết quả xét nghiệm nhanh cho thấy Miu Lê và 3 người dương tính với Methamphetamine, Ketamine và MDMA. Khám xét hiện trường tại bãi tắm Tùng Thu, công an thu giữ ma túy tổng hợp nước vui và Ketamine.",
                "score": 0.88,
                "metadata": {
                    "source": "article_01.md",
                    "title": "Ca sĩ Miu Lê bị bắt với cáo buộc tổ chức sử dụng ma túy",
                    "doc_type": "news",
                    "url": "https://vnexpress.net/ca-si-miu-le-bi-bat-voi-cao-buoc-to-chuc-su-dung-ma-tuy-5074769.html",
                    "chunk_index": 1,
                },
                "retrieval_method": "hybrid",
            },
        ]
        return {
            "answer": "Theo tài liệu báo chí [Nguồn 1], ca sĩ Miu Lê (tên thật: Lê Ánh Nhật) đã bị Cơ quan CSĐT Công an TP Hải Phòng khởi tố và bắt tạm giam về hành vi **'Tổ chức sử dụng trái phép chất ma túy'** tại khu vực bãi tắm Tùng Thu, đảo Cát Bà, Hải Phòng. Kết quả xét nghiệm xác định dương tính với **Methamphetamine, Ketamine và MDMA** [Nguồn 2].",
            "sources": sources[:top_k],
            "retrieval_source": "hybrid",
        }

    # Trường hợp 3: Mức xử phạt hành chính theo Nghị định 144
    if "mức phạt" in q or "phạt tiền" in q or "nghị định 144" in q or "bao nhiêu" in q:
        sources = [
            {
                "id": "nd144-chunk-57",
                "content": "Điều 23. Vi phạm các quy định về phòng, chống và kiểm soát ma túy:\n1. Phạt cảnh cáo hoặc phạt tiền từ 1.000.000 đồng đến 2.000.000 đồng đối với hành vi sử dụng trái phép chất ma túy.\n3. Phạt tiền từ 5.000.000 đồng đến 10.000.000 đồng đối với hành vi trồng các loại cây thuốc phiện, cây cần sa, cây coca, cây khát và các loại cây khác có chứa chất ma túy.",
                "score": 0.95,
                "metadata": {
                    "source": "nghi-dinh-144-2021-nd-cp-xu-phat-vi-pham-hanh-chinh.md",
                    "title": "Nghị định 144/2021/NĐ-CP xử phạt vi phạm hành chính",
                    "doc_type": "legal",
                    "url": "https://vanban.vcci.com.vn/nghi-dinh-1442021nd-cp",
                    "chunk_index": 57,
                },
                "retrieval_method": "hybrid",
            },
            {
                "id": "nd144-chunk-58",
                "content": "Điều 23 khoản 4: Phạt tiền từ 10.000.000 đồng đến 20.000.000 đồng đối với người đứng đầu, người quản lý cơ sở kinh doanh (karaoke, khách sạn...) để xảy ra hoạt động tàng trữ, mua bán, sử dụng trái phép chất ma túy trong khu vực mình quản lý.",
                "score": 0.86,
                "metadata": {
                    "source": "nghi-dinh-144-2021-nd-cp-xu-phat-vi-pham-hanh-chinh.md",
                    "title": "Nghị định 144/2021/NĐ-CP xử phạt vi phạm hành chính",
                    "doc_type": "legal",
                    "url": "https://vanban.vcci.com.vn/nghi-dinh-1442021nd-cp",
                    "chunk_index": 58,
                },
                "retrieval_method": "hybrid",
            },
        ]
        return {
            "answer": "Căn cứ theo **Điều 23 Nghị định số 144/2021/NĐ-CP** [Nguồn 1]:\n- **Hành vi sử dụng trái phép chất ma túy:** Phạt cảnh cáo hoặc phạt tiền từ **1.000.000 đồng đến 2.000.000 đồng**.\n- **Hành vi trồng cây có chứa chất ma túy (cần sa, thuốc phiện,...):** Phạt tiền từ **5.000.000 đồng đến 10.000.000 đồng**.\n- **Người quản lý cơ sở kinh doanh (karaoke, khách sạn) để xảy ra ma túy:** Phạt từ **10.000.000 đồng đến 20.000.000 đồng** kèm tước giấy phép 6-12 tháng [Nguồn 2].",
            "sources": sources[:top_k],
            "retrieval_source": "hybrid",
        }

    # Trường hợp mặc định: Luật Phòng, chống ma túy 2021
    sources = [
        {
            "id": "luat73-chunk-15",
            "content": "Điều 5. Các hành vi bị nghiêm cấm:\n1. Trồng cây có chứa chất ma túy, hướng dẫn trồng cây có chứa chất ma túy.\n2. Nghiên cứu, sản xuất, tàng trữ, vận chuyển, mua bán trái phép chất ma túy...\n5. Sử dụng, tổ chức sử dụng trái phép chất ma túy; cưỡng bức, lôi kéo người khác sử dụng trái phép chất ma túy.",
            "score": 0.91,
            "metadata": {
                "source": "luat-73-2021-qh14-phong-chong-ma-tuy.md",
                "title": "Luật Phòng, chống ma túy số 73/2021/QH14",
                "doc_type": "legal",
                "url": None,
                "chunk_index": 15,
            },
            "retrieval_method": "hybrid",
        },
        {
            "id": "luat73-chunk-23",
            "content": "Điều 23. Quản lý người sử dụng trái phép chất ma túy:\n1. Quản lý người sử dụng trái phép chất ma túy là biện pháp phòng ngừa nhằm giúp người sử dụng không tiếp tục sử dụng trái phép chất ma túy...\n2. Thời hạn quản lý là 01 năm kể từ ngày Chủ tịch UBND cấp xã ra quyết định quản lý.",
            "score": 0.85,
            "metadata": {
                "source": "luat-73-2021-qh14-phong-chong-ma-tuy.md",
                "title": "Luật Phòng, chống ma túy số 73/2021/QH14",
                "doc_type": "legal",
                "url": None,
                "chunk_index": 23,
            },
            "retrieval_method": "hybrid",
        },
    ]
    return {
        "answer": f"Theo quy định của **Luật Phòng, chống ma túy năm 2021 (Luật số 73/2021/QH14)** [Nguồn 1], pháp luật nghiêm cấm triệt để các hành vi: trồng cây có chứa chất ma túy; sản xuất, tàng trữ, vận chuyển, mua bán, sử dụng và tổ chức sử dụng trái phép chất ma túy. Thời hạn quản lý người sử dụng trái phép chất ma túy tại cấp xã là **01 năm** [Nguồn 2].",
        "sources": sources[:top_k],
        "retrieval_source": "hybrid",
    }


def real_generate_with_citation(query: str, top_k: int = 5, mode: str = "Hybrid", threshold: float = 0.3) -> dict:
    """Truy vấn pipeline thật từ các module Task 5, 6, 7, 8, 9, 10."""
    try:
        if "Dense-only" in mode:
            chunks = semantic_search(query, top_k=top_k)
        elif "Lexical-only" in mode:
            chunks = lexical_search(query, top_k=top_k)
        elif "Vectorless" in mode:
            chunks = pageindex_search(query, top_k=top_k)
        else:  # Hybrid (RRF + Fallback)
            chunks = retrieve(query, top_k=top_k, score_threshold=threshold, use_reranking=True)

        if not chunks:
            return {
                "answer": "Tôi không thể xác minh thông tin này từ nguồn dữ liệu hiện có trong hệ thống pháp luật và tin tức về phòng, chống ma túy.",
                "sources": [],
                "retrieval_source": "none",
            }

        reordered = reorder_for_llm(chunks)
        context = format_context(reordered)
        user_message = f"Context:\n{context}\n\nQuestion: {query}"
        answer = call_llm(SYSTEM_PROMPT, user_message)

        retrieval_source = chunks[0].get("retrieval_method", "hybrid")
        return {
            "answer": answer,
            "sources": chunks,
            "retrieval_source": retrieval_source,
        }
    except Exception as e:
        st.warning(f"Lưu ý: Pipeline thật báo lỗi ({e}), chuyển sang dữ liệu mẫu.")
        return mock_generate_with_citation(query, top_k=top_k)


def stream_response(text: str):
    """Hiệu ứng gõ chữ khi hiển thị câu trả lời."""
    for word in text.split(" "):
        yield word + " "
        time.sleep(0.04)


# Khởi tạo Session State
if "messages" not in st.session_state:
    st.session_state.messages = []

if "pending_query" not in st.session_state:
    st.session_state.pending_query = None

if "inspect_index" not in st.session_state:
    st.session_state.inspect_index = None

# ==================== SIDEBAR ====================
with st.sidebar:
    st.title("⚖️ Legal RAG")
    st.caption("K4-L3A | TV4: Đào Ngọc Bình Thiên")
    st.divider()

    st.subheader("⚙️ Cấu hình Retrieval")
    use_real_pipeline = st.toggle("Sử dụng Pipeline thật (Gemini LLM)", value=True, help="Bật để gọi mô hình Gemini và hệ thống truy xuất thật")
    top_k = st.slider("Số lượng chunks (top_k)", min_value=1, max_value=10, value=5, step=1)
    retrieval_mode = st.selectbox(
        "Chế độ Retrieval",
        ["Hybrid (Dense + BM25 + RRF)", "Dense-only (ChromaDB)", "Lexical-only (BM25)", "Vectorless (PageIndex)"],
        index=0,
    )
    score_threshold = st.slider("Ngưỡng Fallback Score", 0.0, 1.0, 0.3, 0.05)

    st.divider()
    st.subheader("💡 Câu hỏi mẫu")
    sample_queries = [
        "Ca sĩ Miu Lê bị bắt vì hành vi gì và ở đâu?",
        "Mức phạt hành chính đối với hành vi sử dụng ma túy?",
        "Hành vi nào bị nghiêm cấm theo Luật Phòng chống ma túy?",
        "Thời tiết Hà Nội hôm nay thế nào? (Thử Out-of-Domain)",
    ]
    for sq in sample_queries:
        if st.button(sq, use_container_width=True, key=f"btn_{sq}"):
            st.session_state.pending_query = sq
            st.rerun()

    st.divider()
    if st.button("🗑️ Xóa lịch sử chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.pending_query = None
        st.session_state.inspect_index = None
        st.rerun()

# ==================== GIAO DIỆN CHÍNH (2 CỘT) ====================
col_chat, col_citation = st.columns([6, 4], gap="large")

# Kiểm tra câu hỏi mới từ user (qua ô input hoặc câu hỏi mẫu)
new_user_prompt = None
if st.session_state.pending_query:
    new_user_prompt = st.session_state.pending_query
    st.session_state.pending_query = None

# ----- CỘT TRÁI: KHUNG CHAT RIÊNG BIỆT (SCROLLABLE) -----
with col_chat:
    st.markdown('<div class="main-title">Hệ thống Trợ lý Pháp lý & Tra cứu Ma túy</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Tra cứu dựa trên Luật số 73/2021/QH14, Nghị định 144/2021, Nghị định 105/2021 và tin tức thực tế.</div>', unsafe_allow_html=True)

    # Khung Chat có thanh cuộn riêng (height=560) cố định, không phải cuộn cả trang
    chat_box = st.container(height=560)
    with chat_box:
        for idx, msg in enumerate(st.session_state.messages):
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if msg["role"] == "assistant":
                    sources = msg.get("sources", [])
                    method = msg.get("retrieval_source", "none")
                    if sources:
                        source_names = list({s["metadata"]["source"] for s in sources})
                        st.caption(f"📌 Nguồn: {', '.join(source_names)} | Phương thức: `{method}`")

        # Nếu có câu hỏi mới cần xử lý, hiển thị và chạy hiệu ứng tìm kiếm ngay trong chat_box
        if new_user_prompt:
            # 1. Thêm user message
            st.session_state.messages.append({"role": "user", "content": new_user_prompt})
            with st.chat_message("user"):
                st.markdown(new_user_prompt)

            # 2. Assistant với Effect đang tìm kiếm
            with st.chat_message("assistant"):
                with st.spinner("🔍 Đang tìm kiếm tài liệu và sinh câu trả lời..."):
                    if use_real_pipeline:
                        gen_result = real_generate_with_citation(
                            new_user_prompt,
                            top_k=top_k,
                            mode=retrieval_mode,
                            threshold=score_threshold,
                        )
                    else:
                        time.sleep(0.5)
                        gen_result = mock_generate_with_citation(new_user_prompt, top_k=top_k)

                # Hiệu ứng gõ chữ (stream effect) cho câu trả lời
                st.write_stream(stream_response(gen_result["answer"]))

                # Lưu vào lịch sử tin nhắn
                assistant_msg = {
                    "role": "assistant",
                    "content": gen_result["answer"],
                    "sources": gen_result["sources"],
                    "retrieval_source": gen_result["retrieval_source"],
                }
                st.session_state.messages.append(assistant_msg)
                st.session_state.inspect_index = len(st.session_state.messages) - 1

                if gen_result["sources"]:
                    source_names = list({s["metadata"]["source"] for s in gen_result["sources"]})
                    st.caption(f"📌 Nguồn: {', '.join(source_names)} | Phương thức: `{gen_result['retrieval_source']}`")

    # Ô nhập chat luôn cố định ngay dưới khung chat
    user_input = st.chat_input("Nhập câu hỏi pháp lý hoặc sự việc cần tra cứu...")
    if user_input:
        st.session_state.pending_query = user_input
        st.rerun()

# ----- CỘT PHẢI: CITATION & SOURCES INSPECTOR (SCROLLABLE RIÊNG) -----
with col_citation:
    st.subheader("📑 Bằng chứng & Trích dẫn (Citations)")
    st.caption("Các đoạn trích tài liệu được RAG Pipeline đối soát làm căn cứ trả lời.")

    # Tìm câu trả lời của trợ lý để hiển thị trích dẫn (mặc định lấy câu gần nhất)
    assistant_messages = [
        (i, m) for i, m in enumerate(st.session_state.messages) if m["role"] == "assistant"
    ]

    if not assistant_messages:
        # Hộp thông báo rỗng
        citation_box = st.container(height=560)
        with citation_box:
            st.info("💡 Chưa có trích dẫn nào.\n\nHãy nhập câu hỏi hoặc chọn câu hỏi mẫu ở cột bên trái. Sau khi hệ thống hoàn tất tra cứu, toàn bộ tài liệu đối chiếu và đoạn trích sẽ tự động hiển thị tại đây.")
    else:
        # Lấy tin nhắn cần soi (mặc định là câu mới nhất)
        if st.session_state.inspect_index is not None and st.session_state.inspect_index < len(st.session_state.messages):
            selected_msg = st.session_state.messages[st.session_state.inspect_index]
        else:
            selected_msg = assistant_messages[-1][1]

        active_sources = selected_msg.get("sources", [])
        active_method = selected_msg.get("retrieval_source", "none")

        citation_box = st.container(height=560)
        with citation_box:
            if not active_sources:
                st.warning("⚠️ **Không có nguồn trích dẫn:**\nCâu hỏi ngoài phạm vi dữ liệu hoặc không tìm thấy bằng chứng phù hợp trong kho tài liệu (Safe Refusal).")
            else:
                st.markdown(
                    f"""
                    <div class="citation-header">
                        <span class="meta-badge badge-method">Phương thức: {active_method.upper()}</span>
                        <span>Tìm thấy <b>{len(active_sources)}</b> đoạn trích liên quan</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                for idx, src in enumerate(active_sources, 1):
                    meta = src.get("metadata", {})
                    doc_type = meta.get("doc_type", "unknown")
                    badge_class = "badge-legal" if doc_type == "legal" else "badge-news"
                    type_label = "Văn bản Luật" if doc_type == "legal" else "Tin tức / Bài báo"
                    score = src.get("score", 0.0)

                    with st.expander(f"📌 [Nguồn {idx}] {meta.get('title', 'Tài liệu')} ({score * 100:.0f}%)", expanded=(idx == 1)):
                        st.markdown(
                            f"""
                            <div>
                                <span class="meta-badge {badge_class}">{type_label}</span>
                                <span class="score-badge">Độ khớp: {score * 100:.1f}%</span>
                            </div>
                            <div style="font-size: 0.8rem; color: #94A3B8; margin: 6px 0;">
                                <b>File:</b> <code>{meta.get('source')}</code> | <b>Chunk:</b> #{meta.get('chunk_index')}
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                        if meta.get("url"):
                            st.markdown(f"🔗 [Xem nguồn gốc trực tuyến]({meta.get('url')})")

                        st.markdown("**Đoạn trích căn cứ:**")
                        st.code(src.get("content", ""), language="text", wrap_lines=True)
