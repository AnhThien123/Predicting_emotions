"""
Ứng dụng Nhận diện Cảm xúc Văn bản Tiếng Việt
Sử dụng mô hình PhoBERT đã fine-tune trên bộ dữ liệu UIT-VSMEC.

Chạy:
    streamlit run app.py

Yêu cầu: thư mục model đã huấn luyện (vd: "phobert-vsmec-emotion-final")
được giải nén và đặt CÙNG THƯ MỤC với file app.py này (xem MODEL_DIR bên dưới).
"""

import streamlit as st
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ============================================================
# CẤU HÌNH
# ============================================================
MODEL_DIR = "phobert-vsmec-emotion-final"   # đổi tên nếu thư mục model của bạn khác
MAX_LENGTH = 128

EMOTION_META = {
    "Enjoyment": {"label_vi": "Vui vẻ",     "emoji": "😄", "color": "#2E9E5B"},
    "Sadness":   {"label_vi": "Buồn",       "emoji": "😢", "color": "#5B7FDE"},
    "Anger":     {"label_vi": "Giận dữ",    "emoji": "😠", "color": "#E4572E"},
    "Fear":      {"label_vi": "Sợ hãi",     "emoji": "😨", "color": "#8E44AD"},
    "Surprise":  {"label_vi": "Ngạc nhiên", "emoji": "😲", "color": "#F2B134"},
    "Disgust":   {"label_vi": "Ghê tởm",    "emoji": "🤢", "color": "#6B8F3B"},
    "Other":     {"label_vi": "Khác",       "emoji": "😐", "color": "#8895AB"},
}

NAVY = "#1B2A4A"
BLUE = "#5B7FDE"
CORAL = "#FF6B4A"
ICEBLUE = "#EAF0FB"
CARDBG = "#F4F7FD"

EXAMPLES = [
    "Hôm nay mình thi đậu học bổng, vui quá trời luôn!",
    "Buồn quá, con mèo của mình mới mất hôm qua.",
    "Sao lại có người vô ý thức như vậy, tức thật sự!",
    "Đi một mình trong hẻm tối làm mình sợ quá.",
    "Ủa sao tự nhiên lại có chuyện này, không ngờ tới luôn.",
    "Nhìn cảnh đó thấy ghê, ăn không nổi luôn.",
]

# ============================================================
# GIAO DIỆN — CSS TÙY CHỈNH
# ============================================================
st.set_page_config(
    page_title="Nhận diện Cảm xúc Tiếng Việt",
    page_icon="🇻🇳",
    layout="centered",
    initial_sidebar_state="expanded",
)

st.markdown(f"""
<style>
    .stApp {{
        background: linear-gradient(180deg, #FAFBFF 0%, #FFFFFF 100%);
    }}
    #MainMenu, footer, header {{visibility: hidden;}}

    .hero {{
        background: linear-gradient(135deg, {NAVY} 0%, #24365C 100%);
        border-radius: 18px;
        padding: 2rem 2.2rem;
        margin-bottom: 1.6rem;
        color: white;
    }}
    .hero h1 {{
        font-size: 1.7rem;
        font-weight: 800;
        margin: 0 0 0.3rem 0;
        color: white;
    }}
    .hero p {{
        color: {ICEBLUE};
        font-size: 0.95rem;
        margin: 0;
        opacity: 0.9;
    }}
    .accent-bar {{
        width: 46px; height: 5px; background: {CORAL};
        border-radius: 3px; margin: 0.7rem 0 0.9rem 0;
    }}

    .result-card {{
        border-radius: 18px;
        padding: 1.8rem;
        text-align: center;
        margin-top: 1.2rem;
        margin-bottom: 1.2rem;
        border: 1px solid #E4E9F5;
        box-shadow: 0 6px 20px rgba(27, 42, 74, 0.08);
    }}
    .result-emoji {{ font-size: 3.4rem; line-height: 1; }}
    .result-label {{ font-size: 1.5rem; font-weight: 800; color: {NAVY}; margin-top: 0.3rem; }}
    .result-sub {{ font-size: 0.95rem; color: #6B7593; margin-top: 0.1rem; }}

    .prob-row {{ display: flex; align-items: center; gap: 0.6rem; margin-bottom: 0.55rem; }}
    .prob-name {{ width: 130px; font-size: 0.88rem; font-weight: 600; color: {NAVY}; }}
    .prob-pct {{ width: 52px; font-size: 0.85rem; font-weight: 700; text-align: right; color: #44506B; }}

    .stButton>button {{
        background: {CORAL}; color: white; border: none; border-radius: 10px;
        padding: 0.6rem 1.4rem; font-weight: 700; font-size: 1rem; width: 100%;
    }}
    .stButton>button:hover {{ background: #e85a3a; color: white; }}

    textarea {{ border-radius: 12px !important; }}

    .footnote {{ font-size: 0.78rem; color: #9AA5BD; text-align: center; margin-top: 2rem; }}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
  <h1>🧠 Nhận diện Cảm xúc Văn bản Tiếng Việt</h1>
  <div class="accent-bar"></div>
  <p>Fine-tuning PhoBERT trên bộ dữ liệu UIT-VSMEC — 7 nhãn cảm xúc: Vui vẻ, Buồn, Giận dữ,
  Sợ hãi, Ngạc nhiên, Ghê tởm, Khác</p>
</div>
""", unsafe_allow_html=True)


# ============================================================
# NẠP MÔ HÌNH (cache để không load lại mỗi lần bấm nút)
# ============================================================
@st.cache_resource(show_spinner=False)
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    model.eval()
    return tokenizer, model


@st.cache_resource(show_spinner=False)
def load_segmenter():
    # underthesea tự tải dữ liệu cần thiết ở lần đầu sử dụng
    from underthesea import word_tokenize
    return word_tokenize


def predict(sentence: str, tokenizer, model, word_tokenize):
    segmented = word_tokenize(sentence, format="text")
    inputs = tokenizer(segmented, return_tensors="pt", truncation=True, max_length=MAX_LENGTH)
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = F.softmax(logits, dim=-1)[0].cpu().numpy()
    id2label = model.config.id2label
    result = {id2label[i]: float(p) for i, p in enumerate(probs)}
    return dict(sorted(result.items(), key=lambda x: x[1], reverse=True))


# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown("### ℹ️ Về ứng dụng")
    st.markdown(
        "Mô hình: **PhoBERT-base** fine-tune trên **UIT-VSMEC**\n\n"
        "6.927 câu bình luận mạng xã hội, 7 nhãn cảm xúc.\n\n"
        "---"
    )
    st.markdown("### 🎭 7 nhãn cảm xúc")
    for key, meta in EMOTION_META.items():
        st.markdown(f"{meta['emoji']} **{meta['label_vi']}** ({key})")
    st.markdown("---")
    st.markdown(
        "<span style='font-size:0.8rem;color:#9AA5BD'>"
        "Lưu ý: mô hình được huấn luyện trên tiếng Việt <b>có dấu</b>. "
        "Nhập câu không dấu có thể cho kết quả kém chính xác hơn."
        "</span>", unsafe_allow_html=True,
    )

# ============================================================
# TẢI MÔ HÌNH — báo lỗi rõ ràng nếu chưa có thư mục model
# ============================================================
import os

if not os.path.isdir(MODEL_DIR):
    st.error(
        f"❌ Không tìm thấy thư mục model **`{MODEL_DIR}`**.\n\n"
        f"Hãy giải nén `phobert-vsmec-emotion-final.zip` sao cho thư mục "
        f"**`{MODEL_DIR}`** nằm cùng cấp với file `app.py` này, rồi chạy lại ứng dụng."
    )
    st.stop()

with st.spinner("Đang nạp mô hình PhoBERT..."):
    tokenizer, model = load_model()
    word_tokenize = load_segmenter()

# ============================================================
# NHẬP CÂU & DỰ ĐOÁN
# ============================================================
st.markdown("#### ✍️ Nhập câu văn bản tiếng Việt")

if "sentence_input" not in st.session_state:
    st.session_state.sentence_input = ""

cols = st.columns(3)
for i, ex in enumerate(EXAMPLES):
    if cols[i % 3].button(f"💬 {ex[:22]}...", key=f"ex_{i}", use_container_width=True):
        st.session_state.sentence_input = ex

sentence = st.text_area(
    label="Câu cần phân tích",
    key="sentence_input",
    placeholder="Ví dụ: Hôm nay mình vui muốn xỉu luôn á!",
    height=100,
    label_visibility="collapsed",
)

analyze = st.button("🔍 Phân tích cảm xúc", use_container_width=True)

if analyze:
    if not sentence.strip():
        st.warning("Vui lòng nhập một câu văn bản trước khi phân tích.")
    else:
        with st.spinner("Đang phân tích..."):
            probs = predict(sentence, tokenizer, model, word_tokenize)

        top_label, top_prob = next(iter(probs.items()))
        meta = EMOTION_META.get(top_label, {"label_vi": top_label, "emoji": "🤔", "color": NAVY})

        st.markdown(f"""
        <div class="result-card" style="background:{CARDBG}; border-top: 5px solid {meta['color']};">
            <div class="result-emoji">{meta['emoji']}</div>
            <div class="result-label">{meta['label_vi']} <span style="color:#9AA5BD;font-weight:500;font-size:1rem;">({top_label})</span></div>
            <div class="result-sub">Độ tin cậy: <b>{top_prob*100:.2f}%</b></div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("##### 📊 Phân phối xác suất theo từng cảm xúc")
        for label, prob in probs.items():
            m = EMOTION_META.get(label, {"label_vi": label, "emoji": "", "color": BLUE})
            pct = prob * 100
            st.markdown(f"""
            <div class="prob-row">
                <div class="prob-name">{m['emoji']} {m['label_vi']}</div>
                <div style="flex:1; background:#EEF1F9; border-radius:6px; height:14px; overflow:hidden;">
                    <div style="width:{pct:.1f}%; background:{m['color']}; height:100%; border-radius:6px;"></div>
                </div>
                <div class="prob-pct">{pct:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)

        with st.expander("Xem câu sau khi tách từ (word segmentation)"):
            st.code(word_tokenize(sentence, format="text"))

st.markdown(
    '<div class="footnote">Đồ án: Mạng học sâu BERT và ứng dụng nhận diện cảm xúc '
    'trong văn bản tiếng Việt · PhoBERT + UIT-VSMEC</div>',
    unsafe_allow_html=True,
)
