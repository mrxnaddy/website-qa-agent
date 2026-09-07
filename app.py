"""
app.py
------
Main Streamlit application for the Website Q&A AI Agent.

Run locally with:
    streamlit run app.py
"""

import streamlit as st
from dotenv import load_dotenv

from scraper import scrape, is_pdf_url, PLAYWRIGHT_AVAILABLE
from llm import get_answer, find_relevant_excerpt

load_dotenv()  # loads GROQ_API_KEY from .env for local development

st.set_page_config(
    page_title="Website Q&A AI Agent",
    page_icon="🌐",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Custom CSS — visual polish only, no functional changes
# ---------------------------------------------------------------------------

st.markdown(
    """
    <style>
    /* ---------- Global font & background ---------- */
    html, body, [class*="css"]  {
        font-family: 'Segoe UI', 'Inter', sans-serif;
    }

    .stApp {
        background: linear-gradient(180deg, #0f172a 0%, #111827 45%, #0f172a 100%);
    }

    /* ---------- Sidebar ---------- */
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
    section[data-testid="stSidebar"] .stTextInput input::placeholder {
        color: #cbd5e1 !important;
    }

    /* ---------- Buttons ---------- */
    .stButton > button {
        border-radius: 10px !important;
        font-weight: 600 !important;
        transition: all 0.2s ease-in-out;
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

    /* ---------- Header ---------- */
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

    /* ---------- Metrics ---------- */
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

    /* ---------- Alerts ---------- */
    div[data-testid="stAlert"] {
        border-radius: 12px !important;
    }

    /* ---------- Expanders ---------- */
    details {
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 12px !important;
        padding: 4px 8px;
    }

    /* ---------- Chat bubbles ---------- */
    div[data-testid="stChatMessage"] {
        border-radius: 16px;
        padding: 6px 4px;
        margin-bottom: 6px;
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.06);
    }

    /* ---------- Chat input ---------- */
    div[data-testid="stChatInput"] textarea {
        border-radius: 12px !important;
        background: rgba(255,255,255,0.05) !important;
    }

    /* ---------- Divider ---------- */
    hr {
        border-color: rgba(255,255,255,0.10) !important;
    }

    /* ---------- Text area (raw scraped text) ---------- */
    textarea[disabled] {
        background: rgba(255,255,255,0.03) !important;
        color: #cbd5e1 !important;
        border-radius: 10px !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Session state initialization
# ---------------------------------------------------------------------------

def init_state():
    defaults = {
        "site_a": None,          # dict from scrape() for primary URL
        "site_b": None,          # dict from scrape() for comparison URL
        "url_a": "",
        "url_b": "",
        "compare_mode": False,
        "messages": [],          # [{"role": "user"/"assistant", "content": str, "excerpt": str|None}]
        "scraped": False,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


init_state()


def reset_session():
    for key in ["site_a", "site_b", "messages", "scraped"]:
        st.session_state[key] = [] if key == "messages" else (False if key == "scraped" else None)
    st.session_state["url_a"] = ""
    st.session_state["url_b"] = ""


# ---------------------------------------------------------------------------
# Sidebar — URL input & controls
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("## 🌐 Website Q&A Agent")
    st.caption("✨ Paste a URL, scrape it, then chat with its content.")
    st.markdown("---")

    st.session_state.compare_mode = st.checkbox(
        "🔁 Compare two websites", value=st.session_state.compare_mode
    )

    url_a_input = st.text_input(
        "🔗 Website URL (or PDF link)",
        value=st.session_state.url_a,
        placeholder="https://example.com",
    )

    url_b_input = ""
    if st.session_state.compare_mode:
        url_b_input = st.text_input(
            "🔗 Second website URL (for comparison)",
            value=st.session_state.url_b,
            placeholder="https://another-example.com",
        )

    scrape_clicked = st.button("🔎 Scrape & Start", type="primary", use_container_width=True)

    st.divider()
    if st.button("🔄 Clear chat / New website", use_container_width=True):
        reset_session()
        st.rerun()

    st.divider()
    with st.expander("ℹ️ About this app"):
        st.write(
            "This agent scrapes a webpage's readable text, then answers your "
            "questions strictly based on that content using Groq's "
            "`llama-3.3-70b-versatile` model. It supports English, Urdu, and "
            "Roman Urdu. It will never make up information that isn't on the page."
        )


# ---------------------------------------------------------------------------
# Handle scraping
# ---------------------------------------------------------------------------

def render_scrape_summary(site: dict, label: str):
    if not site or not site.get("success"):
        return
    cols = st.columns(4)
    cols[0].metric(f"{label} Words", f"{site['word_count']:,}")
    cols[1].metric(f"{label} Characters", f"{site['char_count']:,}")
    cols[2].metric(f"{label} Language", site.get("language") or "unknown")
    method_label = {
        "trafilatura": "Trafilatura",
        "beautifulsoup": "BeautifulSoup (fallback)",
        "pdfplumber": "pdfplumber (PDF)",
    }.get(site.get("method"), site.get("method") or "—")
    cols[3].metric(f"{label} Method", method_label)


if scrape_clicked:
    if not url_a_input.strip():
        st.sidebar.error("Please enter a website URL first.")
    else:
        st.session_state.url_a = url_a_input.strip()
        st.session_state.url_b = url_b_input.strip() if st.session_state.compare_mode else ""

        with st.spinner(f"Scraping {st.session_state.url_a} ..."):
            result_a = scrape(st.session_state.url_a)
        st.session_state.site_a = result_a

        if st.session_state.compare_mode and st.session_state.url_b:
            with st.spinner(f"Scraping {st.session_state.url_b} ..."):
                result_b = scrape(st.session_state.url_b)
            st.session_state.site_b = result_b
        else:
            st.session_state.site_b = None

        if result_a["success"]:
            st.session_state.scraped = True
            st.session_state.messages = []  # fresh conversation for new content
        else:
            st.session_state.scraped = False


# ---------------------------------------------------------------------------
# Main area
# ---------------------------------------------------------------------------

st.markdown('<div class="hero-title">🌐 Website Q&A AI Agent</div>', unsafe_allow_html=True)
st.markdown(
    '<p class="hero-subtitle">Scrape any webpage or PDF and chat with its content — in English, Urdu, or Roman Urdu.</p>',
    unsafe_allow_html=True,
)

if not st.session_state.scraped:
    st.info(
        "👈 Paste a website URL (or a direct PDF link) in the sidebar and click "
        "**Scrape & Start** to begin. In compare mode, add a second URL too."
    )
    # Show scraping error if the last attempt failed
    if st.session_state.site_a and not st.session_state.site_a.get("success"):
        st.error(f"⚠️ {st.session_state.site_a['error']}")
    if st.session_state.site_b and not st.session_state.site_b.get("success"):
        st.error(f"⚠️ (Second URL) {st.session_state.site_b['error']}")
    st.stop()

# --- Successfully scraped: show summary -------------------------------------
site_a = st.session_state.site_a
site_b = st.session_state.site_b

kind_a = "PDF" if is_pdf_url(st.session_state.url_a) else "Page"
st.success(f"✅ Scraped **{kind_a}**: {site_a.get('title') or st.session_state.url_a}")
render_scrape_summary(site_a, "A —")

if site_a.get("is_low_content"):
    if PLAYWRIGHT_AVAILABLE:
        st.warning(
            "⚠️ Very little text was found on this page, even after trying a "
            "headless-browser render. It may still be missing dynamic content "
            "loaded after the page finishes rendering, or the page may genuinely "
            "be this short. Answers may be incomplete."
        )
    else:
        st.warning(
            "⚠️ Very little text was extracted — this site is likely "
            "JavaScript-heavy (a React/Next.js/Vue app), so much of its content "
            "loads client-side and isn't visible to a plain HTTP scraper. "
            "For better results on sites like this, install Playwright for a "
            "headless-browser fallback:\n\n"
            "```\npip install playwright\nplaywright install chromium\n```\n"
            "Then restart the app — no code changes needed."
        )

if st.session_state.compare_mode and site_b:
    if site_b.get("success"):
        kind_b = "PDF" if is_pdf_url(st.session_state.url_b) else "Page"
        st.success(f"✅ Scraped **{kind_b}**: {site_b.get('title') or st.session_state.url_b}")
        render_scrape_summary(site_b, "B —")
    else:
        st.warning(f"⚠️ Second URL couldn't be scraped: {site_b['error']}. Continuing in single-site mode.")

with st.expander("📄 View raw scraped text (Website A)"):
    st.text_area("Scraped content", site_a["text"], height=250, disabled=True, label_visibility="collapsed")

if st.session_state.compare_mode and site_b and site_b.get("success"):
    with st.expander("📄 View raw scraped text (Website B)"):
        st.text_area("Scraped content B", site_b["text"], height=250, disabled=True, label_visibility="collapsed")

st.divider()

# ---------------------------------------------------------------------------
# Chat interface
# ---------------------------------------------------------------------------

st.subheader("💬 Ask questions about this content")
st.caption("You can ask in English, Urdu, or Roman Urdu. Try: \"summarize this\" / \"is website ka summary do\"")

for msg in st.session_state.messages:
    avatar = "🧑‍💻" if msg["role"] == "user" else "🤖"
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("excerpt"):
            with st.expander("🔍 Source excerpt (likely basis for this answer)"):
                st.write(msg["excerpt"])

user_question = st.chat_input("Type your question here...")

if user_question:
    st.session_state.messages.append({"role": "user", "content": user_question, "excerpt": None})
    with st.chat_message("user", avatar="🧑‍💻"):
        st.markdown(user_question)

    # Build history for the LLM (role/content only)
    history_for_llm = [
        {"role": m["role"], "content": m["content"]} for m in st.session_state.messages
    ]

    content_b_text = None
    title_b = None
    url_b_final = None
    if st.session_state.compare_mode and site_b and site_b.get("success"):
        content_b_text = site_b["text"]
        title_b = site_b.get("title")
        url_b_final = st.session_state.url_b

    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("Thinking..."):
            result = get_answer(
                conversation_history=history_for_llm,
                content=site_a["text"],
                content_b=content_b_text,
                title=site_a.get("title"),
                title_b=title_b,
                url=st.session_state.url_a,
                url_b=url_b_final,
            )

        if result["success"]:
            answer = result["answer"]
            st.markdown(answer)

            excerpt = find_relevant_excerpt(user_question, site_a["text"])
            if excerpt:
                with st.expander("🔍 Source excerpt (likely basis for this answer)"):
                    st.write(excerpt)

            st.session_state.messages.append(
                {"role": "assistant", "content": answer, "excerpt": excerpt}
            )
        else:
            error_text = f"⚠️ {result['error']}"
            st.error(error_text)
            st.session_state.messages.append(
                {"role": "assistant", "content": error_text, "excerpt": None}
            )