"""
app.py
------
Main Streamlit application for the Website / Document Q&A AI Agent.

Run:
    streamlit run app.py
"""

import shutil
#import pytesseract
try:
    import pytesseract
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False
import streamlit as st
from dotenv import load_dotenv

from scraper import scrape, scrape_uploaded_file, is_pdf_url, PLAYWRIGHT_AVAILABLE
from llm import get_answer, find_relevant_excerpt

# # Check if running on Streamlit Cloud (Linux) or Windows
# if shutil.which("tesseract"):
#     # Linux / Streamlit Cloud finds it automatically via PATH
#     pass
# else:
#     # Local Windows fallback path
#     try:
#         pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
#     except Exception:
#         pass

load_dotenv()

st.set_page_config(
    page_title="Source Q&A AI Agent",
    page_icon="🤖",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------

st.markdown(
    """
    <style>
    html, body, [class*="css"] {
        font-family: 'Segoe UI', 'Inter', sans-serif;
    }
    .stApp {
        background: linear-gradient(180deg, #0f172a 0%, #111827 45%, #0f172a 100%);
    }
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1e1b4b 0%, #312e81 100%);
        border-right: 1px solid rgba(255,255,255,0.08);
    }
    section[data-testid="stSidebar"] * {
        color: #f1f5f9 !important;
    }
    section[data-testid="stSidebar"] .stTextInput input {
        background-color: rgba(255,255,255,0.08);
        border: 1px solid rgba(255,255,255,0.15);
        border-radius: 10px;
        color: #f8fafc !important;
    }
    .stButton > button {
        border-radius: 10px !important;
        font-weight: 600 !important;
        border: none !important;
    }
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 6px 14px rgba(99, 102, 241, 0.35);
    }
    button[kind="primary"] {
        background: linear-gradient(90deg, #6366f1, #8b5cf6) !important;
        color: white !important;
    }
    .hero-title {
        font-size: 2.4rem;
        font-weight: 800;
        background: linear-gradient(90deg, #a5b4fc, #f0abfc, #f472b6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .hero-subtitle {
        color: #cbd5e1;
        font-size: 1.02rem;
        margin-top: 0;
        margin-bottom: 1.2rem;
    }
    div[data-testid="stMetric"] {
        background: rgba(99, 102, 241, 0.10);
        border: 1px solid rgba(99, 102, 241, 0.35);
        border-radius: 14px;
        padding: 12px 10px;
        text-align: center;
    }
    div[data-testid="stMetricLabel"] {
        color: #c7d2fe !important;
    }
    div[data-testid="stMetricValue"] {
        color: #f8fafc !important;
    }
    div[data-testid="stChatMessage"] {
        border-radius: 16px;
        padding: 6px 4px;
        margin-bottom: 6px;
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.06);
    }
    div[data-testid="stChatInput"] textarea {
        border-radius: 12px !important;
        background: rgba(255,255,255,0.05) !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

def init_state():
    defaults = {
        "site_a": None,
        "site_b": None,
        "url_a": "",
        "url_b": "",
        "compare_mode": False,
        "messages": [],
        "scraped": False,
        "content_source": None,
        "source_name": None,
        "image_bytes": None,
        "image_mime": "image/jpeg",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


init_state()


def reset_session():
    for key in [
        "site_a", "site_b", "messages", "scraped",
        "content_source", "source_name", "image_bytes"
    ]:
        if key == "messages":
            st.session_state[key] = []
        elif key == "scraped":
            st.session_state[key] = False
        else:
            st.session_state[key] = None

    st.session_state["image_mime"] = "image/jpeg"
    st.session_state["url_a"] = ""
    st.session_state["url_b"] = ""
    st.session_state["compare_mode"] = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def render_scrape_summary(site: dict, label: str):
    if not site or not site.get("success"):
        return

    cols = st.columns(5)
    cols[0].metric(f"{label} Words", f"{site.get('word_count', 0):,}")
    cols[1].metric(f"{label} Characters", f"{site.get('char_count', 0):,}")
    cols[2].metric(f"{label} Language", site.get("language") or "unknown")

    method_label = {
        "trafilatura": "Trafilatura",
        "beautifulsoup": "BeautifulSoup",
        "beautifulsoup (broad)": "BeautifulSoup",
        "pdfplumber": "PDF text",
        "pdfplumber (upload)": "PDF text",
        "PyMuPDF + Tesseract OCR": "PDF OCR",
        "PIL + Tesseract OCR": "Image OCR",
        "python-docx (upload)": "DOCX",
        "pandas (Excel upload)": "Excel",
        "pandas (CSV upload)": "CSV",
        "python-pptx (upload)": "PowerPoint",
    }.get(site.get("method"), site.get("method") or "—")

    cols[3].metric(f"{label} Method", method_label)
    cols[4].metric(
        f"{label} OCR",
        "Yes" if site.get("ocr_used") else "No",
    )


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("## 🤖 Source Q&A Agent")
    st.caption("Websites, documents, scans and images → grounded AI answers.")
    st.markdown("---")

    input_mode = st.radio(
        "📥 Content source",
        ["🌐 Website / URL", "📎 Upload file"],
        horizontal=True,
        label_visibility="collapsed",
    )

    url_a_input = ""
    url_b_input = ""
    uploaded_file = None

    if input_mode == "🌐 Website / URL":
        st.session_state.compare_mode = st.checkbox(
            "🔁 Compare two websites",
            value=st.session_state.compare_mode,
        )

        url_a_input = st.text_input(
            "🔗 Website URL (or PDF link)",
            value=st.session_state.url_a,
            placeholder="https://example.com",
        )

        if st.session_state.compare_mode:
            url_b_input = st.text_input(
                "🔗 Second website URL",
                value=st.session_state.url_b,
                placeholder="https://another-example.com",
            )

    else:
        st.session_state.compare_mode = False

        uploaded_file = st.file_uploader(
            "📎 Upload source",
            type=[
                "pdf", "docx", "txt", "md",
                "xlsx", "xls", "csv", "pptx",
                "png", "jpg", "jpeg", "webp", "bmp", "tif", "tiff",
            ],
            help=(
                "PDF, scanned PDF, Word, Excel, CSV, PowerPoint and common "
                "image formats are supported."
            ),
        )

        st.caption(
            "PDF • Scanned PDF • DOCX • TXT • MD • XLSX • XLS • CSV • PPTX • "
            "PNG • JPG • WEBP • TIFF"
        )

    analyze_clicked = st.button(
        "🚀 Analyze & Start",
        type="primary",
        use_container_width=True,
    )

    st.divider()

    if st.button("🔄 Clear / New source", use_container_width=True):
        reset_session()
        st.rerun()

    st.divider()

    with st.expander("ℹ️ How it works"):
        st.write(
            "The app extracts text from your source, uses OCR when necessary, "
            "retrieves the most relevant source sections for each question, and "
            "asks Groq to answer from that source. Uploaded images can also be "
            "sent directly to a Groq vision model."
        )


# ---------------------------------------------------------------------------
# Analyze source
# ---------------------------------------------------------------------------

if analyze_clicked:
    st.session_state.site_a = None
    st.session_state.site_b = None
    st.session_state.messages = []
    st.session_state.scraped = False
    st.session_state.content_source = None
    st.session_state.source_name = None
    st.session_state.image_bytes = None

    if input_mode == "📎 Upload file":
        if uploaded_file is None:
            st.sidebar.error("Please choose a file first.")
        else:
            file_bytes = uploaded_file.getvalue()
            filename = uploaded_file.name

            with st.spinner(f"Analyzing {filename} ..."):
                result_a = scrape_uploaded_file(file_bytes, filename)

            st.session_state.site_a = result_a
            st.session_state.content_source = "file"
            st.session_state.source_name = filename
            st.session_state.url_a = ""

            ext = filename.lower().rsplit(".", 1)[-1] if "." in filename else ""
            if ext in {"png", "jpg", "jpeg", "webp", "bmp", "tif", "tiff"}:
                if not OCR_AVAILABLE:
                    st.warning("⚠️ Scanned images and OCR features are currently unavailable on this cloud deployment.")
                else:
                    mime_map = {
                        "png": "image/png",
                        "jpg": "image/jpeg",
                        "jpeg": "image/jpeg",
                        "webp": "image/webp",
                        "bmp": "image/bmp",
                        "tif": "image/tiff",
                        "tiff": "image/tiff",
                    }
                    # Keep original image for Groq vision analysis.
                    st.session_state.image_bytes = file_bytes
                    st.session_state.image_mime = mime_map.get(ext, "image/jpeg")

            if result_a.get("success"):
                st.session_state.scraped = True

    else:
        if not url_a_input.strip():
            st.sidebar.error("Please enter a website URL first.")
        else:
            st.session_state.url_a = url_a_input.strip()
            st.session_state.url_b = (
                url_b_input.strip() if st.session_state.compare_mode else ""
            )

            with st.spinner(f"Scraping {st.session_state.url_a} ..."):
                result_a = scrape(st.session_state.url_a)

            st.session_state.site_a = result_a
            st.session_state.content_source = "url"
            st.session_state.source_name = st.session_state.url_a

            if st.session_state.compare_mode and st.session_state.url_b:
                with st.spinner(f"Scraping {st.session_state.url_b} ..."):
                    result_b = scrape(st.session_state.url_b)
                st.session_state.site_b = result_b

            if result_a.get("success"):
                st.session_state.scraped = True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

st.markdown(
    '<div class="hero-title">🤖 Source Q&A AI Agent</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<p class="hero-subtitle">Analyze websites, PDFs, scans, documents, spreadsheets, presentations and images — then ask questions in English, Urdu or Roman Urdu.</p>',
    unsafe_allow_html=True,
)

if not st.session_state.scraped:
    st.info(
        "👈 Add a website/PDF URL or upload a file, then click "
        "**Analyze & Start**."
    )

    if st.session_state.site_a and not st.session_state.site_a.get("success"):
        st.error(f"⚠️ {st.session_state.site_a.get('error', 'Unknown error')}")

    if st.session_state.site_b and not st.session_state.site_b.get("success"):
        st.error(f"⚠️ Second source: {st.session_state.site_b.get('error', 'Unknown error')}")

    st.stop()


site_a = st.session_state.site_a
site_b = st.session_state.site_b

source_kind = "File"
if st.session_state.content_source == "url":
    source_kind = "PDF" if is_pdf_url(st.session_state.url_a) else "Website"

source_label = (
    st.session_state.source_name
    or site_a.get("title")
    or "Source"
)

st.success(f"✅ Analyzed **{source_kind}**: {source_label}")
render_scrape_summary(site_a, "A —")

if site_a.get("ocr_used"):
    st.info(
        "🔎 OCR was used for this source. The extracted OCR text is available "
        "below and is also used for Q&A."
    )

if st.session_state.content_source == "url" and site_a.get("is_low_content"):
    if not PLAYWRIGHT_AVAILABLE:
        st.warning(
            "Very little text was extracted. If the site is JavaScript-heavy, "
            "install Playwright with `pip install playwright` and "
            "`playwright install chromium`."
        )
    else:
        st.warning(
            "Very little text was extracted from this page. Some dynamic content "
            "may still be unavailable."
        )

if st.session_state.image_bytes:
    st.image(
        st.session_state.image_bytes,
        caption="Original image sent to the vision model for visual questions",
        use_container_width=True,
    )

if st.session_state.compare_mode and site_b:
    if site_b.get("success"):
        kind_b = "PDF" if is_pdf_url(st.session_state.url_b) else "Website"
        st.success(
            f"✅ Analyzed **{kind_b}**: "
            f"{site_b.get('title') or st.session_state.url_b}"
        )
        render_scrape_summary(site_b, "B —")
    else:
        st.warning(
            f"⚠️ Second URL couldn't be analyzed: {site_b.get('error')}. "
            "Continuing with source A."
        )

with st.expander("📄 View extracted source text"):
    st.text_area(
        "Extracted text",
        site_a.get("text", ""),
        height=300,
        disabled=True,
        label_visibility="collapsed",
    )

if st.session_state.compare_mode and site_b and site_b.get("success"):
    with st.expander("📄 View extracted text — Source B"):
        st.text_area(
            "Extracted text B",
            site_b.get("text", ""),
            height=300,
            disabled=True,
            label_visibility="collapsed",
        )

st.divider()

# ---------------------------------------------------------------------------
# Chat
# ---------------------------------------------------------------------------

st.subheader("💬 Ask questions about your source")
st.caption(
    'Examples: "summarize this", "what is the deadline?", '
    '"is document mein total fee kitni hai?", "image mein kya likha hai?"'
)

for msg in st.session_state.messages:
    avatar = "🧑‍💻" if msg["role"] == "user" else "🤖"
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("excerpt"):
            with st.expander("🔍 Source excerpt"):
                st.write(msg["excerpt"])


user_question = st.chat_input("Ask anything about the analyzed source...")

if user_question:
    st.session_state.messages.append(
        {"role": "user", "content": user_question, "excerpt": None}
    )

    with st.chat_message("user", avatar="🧑‍💻"):
        st.markdown(user_question)

    history_for_llm = [
        {"role": m["role"], "content": m["content"]}
        for m in st.session_state.messages
    ]

    content_b_text = None
    title_b = None
    url_b_final = None

    if (
        st.session_state.compare_mode
        and site_b
        and site_b.get("success")
    ):
        content_b_text = site_b.get("text", "")
        title_b = site_b.get("title")
        url_b_final = st.session_state.url_b

    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("Reading the relevant source sections..."):
            result = get_answer(
                conversation_history=history_for_llm,
                content=site_a.get("text", ""),
                content_b=content_b_text,
                title=site_a.get("title") or st.session_state.source_name,
                title_b=title_b,
                url=(
                    st.session_state.url_a
                    if st.session_state.content_source == "url"
                    else None
                ),
                url_b=url_b_final,
                image_bytes=st.session_state.image_bytes,
                image_mime=st.session_state.image_mime,
            )

        if result["success"]:
            answer = result["answer"]
            st.markdown(answer)

            excerpt = find_relevant_excerpt(
                user_question,
                site_a.get("text", ""),
            )

            if excerpt:
                with st.expander("🔍 Source excerpt"):
                    st.write(excerpt)

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": answer,
                    "excerpt": excerpt,
                }
            )
        else:
            error_text = f"⚠️ {result['error']}"
            st.error(error_text)
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": error_text,
                    "excerpt": None,
                }
            )
