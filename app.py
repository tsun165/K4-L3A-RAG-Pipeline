import os
import time
import streamlit as st
from dotenv import load_dotenv

from src.task10_generation import generate_with_citation

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


def stream_response(text: str):
    """Hiệu ứng gõ chữ khi hiển thị câu trả lời."""
    for word in str(text).split(" "):
        yield word + " "
        time.sleep(0.03)


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
    top_k = st.slider("Số lượng chunks (top_k)", min_value=1, max_value=10, value=5, step=1)
    retrieval_mode = st.selectbox(
        "Chế độ Retrieval",
        ["Hybrid (Dense + BM25 + RRF)", "Dense-only (ChromaDB)", "Lexical-only (BM25)", "Vectorless (PageIndex)"],
        index=0,
    )
    score_threshold = st.slider("Ngưỡng Fallback Score", 0.0, 1.0, 0.30, 0.05)

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
    st.markdown('<div class="sub-title">Pipeline RAG thật tích hợp Dense Vector, BM25 Okapi, RRF Fusion và PageIndex Fallback.</div>', unsafe_allow_html=True)

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

        # Nếu có câu hỏi mới cần xử lý, chạy pipeline thật ngay trong chat_box
        if new_user_prompt:
            # 1. Thêm user message
            st.session_state.messages.append({"role": "user", "content": new_user_prompt})
            with st.chat_message("user"):
                st.markdown(new_user_prompt)

            # 2. Assistant gọi trực tiếp pipeline thật (src.task10_generation)
            with st.chat_message("assistant"):
                with st.spinner("🔍 Đang truy vấn kho dữ liệu và tổng hợp câu trả lời..."):
                    gen_result = generate_with_citation(new_user_prompt, top_k=top_k)

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

    assistant_messages = [
        (i, m) for i, m in enumerate(st.session_state.messages) if m["role"] == "assistant"
    ]

    citation_box = st.container(height=560)
    with citation_box:
        if not assistant_messages:
            st.info("💡 Chưa có trích dẫn nào.\n\nHãy nhập câu hỏi hoặc chọn câu hỏi mẫu ở cột bên trái. Toàn bộ tài liệu đối chiếu và đoạn trích từ pipeline thật sẽ tự động hiển thị tại đây.")
        else:
            if st.session_state.inspect_index is not None and st.session_state.inspect_index < len(st.session_state.messages):
                selected_msg = st.session_state.messages[st.session_state.inspect_index]
            else:
                selected_msg = assistant_messages[-1][1]

            active_sources = selected_msg.get("sources", [])
            active_method = selected_msg.get("retrieval_source", "none")

            if not active_sources:
                st.warning("⚠️ **Không có nguồn trích dẫn:**\nCâu hỏi ngoài phạm vi dữ liệu hoặc hệ thống thực hiện từ chối an toàn (*Safe Refusal*).")
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
