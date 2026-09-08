"""
scraper.py
----------
Handles all content-extraction logic for the Website Q&A AI Agent.

Responsibilities:
  * Fetch a web page safely (proper headers, timeouts, error handling)
  * Extract clean readable text using trafilatura, with a BeautifulSoup
    fallback if trafilatura returns little/no content
  * Detect and extract text from PDF links using pdfplumber
  * Detect the language of the scraped content
  * Never raise raw exceptions up to the UI layer — always return a
    structured result dict so the caller can show a friendly message.
"""

from __future__ import annotations

import io
import re
import zipfile
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup

try:
    import trafilatura
except ImportError:  # pragma: no cover
    trafilatura = None

try:
    import pdfplumber
except ImportError:  # pragma: no cover
    pdfplumber = None

try:
    import fitz  # PyMuPDF: renders scanned PDF pages for OCR
except ImportError:  # pragma: no cover
    fitz = None

try:
    import pytesseract
except ImportError:  # pragma: no cover
    pytesseract = None

try:
    from PIL import Image
except ImportError:  # pragma: no cover
    Image = None

try:
    from langdetect import detect, LangDetectException
except ImportError:  # pragma: no cover
    detect = None
    LangDetectException = Exception

try:
    import pandas as pd
except ImportError:  # pragma: no cover
    pd = None

try:
    from playwright.sync_api import sync_playwright
    PLAYWRIGHT_AVAILABLE = True
except ImportError:  # pragma: no cover
    PLAYWRIGHT_AVAILABLE = False

# Below this many words in the initial static-HTML extraction, we suspect the
# page is JavaScript-rendered and (if Playwright is installed) attempt a
# headless-browser render as a last-resort fallback.
LOW_CONTENT_WORD_THRESHOLD = 60


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}

REQUEST_TIMEOUT = 15  # seconds
MIN_TRAFILATURA_CHARS = 200  # below this, fall back to BeautifulSoup

# Tags we strip entirely before falling back to BeautifulSoup extraction
TAGS_TO_REMOVE = ["script", "style", "nav", "footer", "header", "aside", "form", "noscript"]

# Tags we consider "readable content" for the BS4 fallback
CONTENT_TAGS = ["h1", "h2", "h3", "p", "li"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def is_valid_url(url: str) -> bool:
    """Very light validation — just checks scheme + netloc are present."""
    try:
        parsed = urlparse(url.strip())
        return all([parsed.scheme in ("http", "https"), parsed.netloc])
    except Exception:
        return False


def is_pdf_url(url: str) -> bool:
    """Detect whether a URL points to a PDF, by extension or content-type sniff."""
    if url.lower().split("?")[0].endswith(".pdf"):
        return True
    try:
        resp = requests.head(url, headers=DEFAULT_HEADERS, timeout=REQUEST_TIMEOUT, allow_redirects=True)
        content_type = resp.headers.get("Content-Type", "").lower()
        return "application/pdf" in content_type
    except requests.RequestException:
        # If HEAD fails, don't block — GET-based scraping will surface the real error.
        return False


def detect_language(text: str) -> str:
    """
    Return a human-readable language label detected from the text.
    Falls back to a Unicode-script heuristic if langdetect is unavailable,
    fails, or the text is too short/noisy for it to work reliably.
    """
    if not text or not text.strip():
        return "unknown"

    cleaned = text.strip()

    # Try langdetect first (works well on real prose of reasonable length)
    if detect is not None and len(cleaned) >= 20:
        try:
            code = detect(cleaned[:3000])
            labels = {
                "en": "English",
                "ur": "Urdu",
                "hi": "Hindi",
                "ar": "Arabic",
                "fr": "French",
                "es": "Spanish",
                "de": "German",
                "zh-cn": "Chinese",
                "ru": "Russian",
            }
            if code in labels:
                return labels[code]
        except Exception:
            pass  # fall through to script-based heuristic below

    # Fallback heuristic: check which Unicode script dominates the text.
    # Useful when langdetect isn't installed, fails, or the sample is too
    # short/mixed (e.g. thin scraped content) for statistical detection.
    arabic_script_chars = len(re.findall(r"[\u0600-\u06FF]", cleaned))
    latin_chars = len(re.findall(r"[A-Za-z]", cleaned))

    if arabic_script_chars > latin_chars and arabic_script_chars > 5:
        return "Urdu"  # (or Arabic — Urdu script overlaps with Arabic Unicode block)
    if latin_chars > 5:
        return "English"

    return "unknown"


def _clean_whitespace(text: str) -> str:
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _extract_embedded_state_text(html: str, min_string_len: int = 12) -> str:
    """
    Modern frameworks (Next.js, Nuxt, Remix, Gatsby, etc.) frequently embed the
    FULL page data as a JSON blob directly in the HTML — before any JS runs —
    to "hydrate" the page client-side. This means real content (event details,
    dates, descriptions) is often present in the static HTML even on sites that
    LOOK javascript-only, just inside a <script> tag instead of visible markup.

    This walks any large JSON <script> blocks (id="__NEXT_DATA__", or generic
    type="application/json") and pulls out string values that look like real
    sentences/labels/dates rather than internal keys, class names, or hashes.
    """
    import json

    soup = BeautifulSoup(html, "html.parser")
    candidates = soup.find_all("script", id="__NEXT_DATA__")
    candidates += soup.find_all("script", type="application/json")

    if not candidates:
        return ""

    collected: list[str] = []
    seen = set()

    # Patterns that indicate "noise" (ids, hashes, css classes, urls, code)
    noise_pattern = re.compile(r"^(https?://|/_next/|#[0-9a-fA-F]{3,8}$|[0-9a-f]{16,}$)")

    def walk(node):
        if isinstance(node, dict):
            for v in node.values():
                walk(v)
        elif isinstance(node, list):
            for v in node:
                walk(v)
        elif isinstance(node, str):
            value = node.strip()
            if (
                len(value) >= min_string_len
                and " " in value  # real phrases have spaces; keys/slugs usually don't
                and not noise_pattern.match(value)
                and value not in seen
            ):
                seen.add(value)
                collected.append(value)

    for script in candidates:
        raw = script.string
        if not raw:
            continue
        try:
            data = json.loads(raw)
        except Exception:
            continue
        walk(data)

    if not collected:
        return ""

    # Cap how much we pull in — this is a best-effort supplement, not the
    # primary content source, so keep it bounded.
    collected = collected[:150]
    return "Additional page data (from embedded app state):\n" + "\n".join(collected)


def _render_with_playwright(url: str, timeout_ms: int = 20000) -> str | None:
    """
    Last-resort fallback for JavaScript-heavy pages: render the page in a
    real (headless) browser and return the fully-rendered HTML. Only runs if
    the `playwright` package AND its browser binaries are installed locally
    (run `playwright install chromium` once after `pip install playwright`).
    Returns None on any failure so callers can fall back gracefully.
    """
    if not PLAYWRIGHT_AVAILABLE:
        return None

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page(user_agent=DEFAULT_HEADERS["User-Agent"])
            page.goto(url, timeout=timeout_ms, wait_until="networkidle")
            html = page.content()
            browser.close()
            return html
    except Exception:
        return None


def _extract_json_ld_data(html: str) -> str:
    """
    Many modern sites embed structured metadata in <script type="application/ld+json">
    blocks (schema.org Event, Product, Article, FAQ, etc.) — this is often the most
    reliable place to find dates, prices, and other facts, since it's frequently
    dropped by both trafilatura and body-text extraction (it isn't "visible" text).
    This pulls out human-readable key facts from any JSON-LD blocks found.
    """
    import json

    soup = BeautifulSoup(html, "html.parser")
    scripts = soup.find_all("script", type="application/ld+json")
    if not scripts:
        return ""

    # Fields worth surfacing when present, in a readable order
    interesting_keys = [
        "name", "headline", "description", "startDate", "endDate",
        "datePublished", "dateModified", "price", "priceCurrency",
        "location", "author", "organizer",
    ]

    lines = []
    for script in scripts:
        raw = script.string
        if not raw:
            continue
        try:
            data = json.loads(raw)
        except Exception:
            continue

        candidates = data if isinstance(data, list) else [data]
        for item in candidates:
            if not isinstance(item, dict):
                continue
            facts = []
            for key in interesting_keys:
                value = item.get(key)
                if value is None:
                    continue
                if isinstance(value, dict):
                    value = value.get("name") or value.get("@id") or str(value)
                if isinstance(value, (str, int, float)) and str(value).strip():
                    facts.append(f"{key}: {value}")
            if facts:
                lines.append(" | ".join(facts))

    if not lines:
        return ""

    return "Structured page data (dates/facts embedded in the page):\n" + "\n".join(lines)


def _bs4_fallback_extract(html: str) -> str:
    """Extract readable text using BeautifulSoup as a fallback strategy."""
    soup = BeautifulSoup(html, "html.parser")

    for tag_name in TAGS_TO_REMOVE:
        for tag in soup.find_all(tag_name):
            tag.decompose()

    chunks = []
    for tag in soup.find_all(CONTENT_TAGS):
        text = tag.get_text(separator=" ", strip=True)
        if text and len(text) > 1:
            chunks.append(text)

    return _clean_whitespace("\n".join(chunks))


def _bs4_last_resort_extract(html: str) -> str:
    """
    Broadest possible fallback: some sites put visible text directly inside
    <div>/<span>/<a> rather than <p>/<li> (common with page builders and
    lightly-templated sites). This grabs ALL visible text from <body> and
    filters out short boilerplate-looking lines (menus, single words, etc.)
    """
    soup = BeautifulSoup(html, "html.parser")

    for tag_name in TAGS_TO_REMOVE:
        for tag in soup.find_all(tag_name):
            tag.decompose()

    body = soup.body or soup
    raw_text = body.get_text(separator="\n", strip=True)

    lines = [line.strip() for line in raw_text.split("\n") if line.strip()]
    # Keep lines that look like actual sentences/phrases, not single nav words.
    # Also keep short lines if there are very few lines overall (tiny pages).
    meaningful = [line for line in lines if len(line) > 25 or " " in line]
    if not meaningful:
        meaningful = lines

    # De-duplicate consecutive repeated lines (common in broken markup)
    deduped = []
    for line in meaningful:
        if not deduped or deduped[-1] != line:
            deduped.append(line)

    return _clean_whitespace("\n".join(deduped))


# ---------------------------------------------------------------------------
# Public: HTML scraping
# ---------------------------------------------------------------------------

def scrape_website(url: str) -> dict:
    """
    Scrape a web page and return clean readable text.

    Returns a dict:
        {
            "success": bool,
            "text": str,          # populated only on success
            "error": str | None,  # populated only on failure
            "method": str,        # "trafilatura" | "beautifulsoup"
            "title": str | None,
        }
    """
    url = url.strip()

    if not is_valid_url(url):
        return {
            "success": False,
            "text": "",
            "error": (
                "That doesn't look like a valid URL. Please include the scheme, "
                "e.g. https://example.com"
            ),
            "method": None,
            "title": None,
        }

    # --- Fetch raw HTML ---------------------------------------------------
    try:
        response = requests.get(
            url, headers=DEFAULT_HEADERS, timeout=REQUEST_TIMEOUT, allow_redirects=True
        )
    except requests.exceptions.Timeout:
        return {
            "success": False,
            "text": "",
            "error": "The website took too long to respond (timeout). Please try again or check the URL.",
            "method": None,
            "title": None,
        }
    except requests.exceptions.ConnectionError:
        return {
            "success": False,
            "text": "",
            "error": "Could not connect to that website. Please check the URL and your internet connection.",
            "method": None,
            "title": None,
        }
    except requests.exceptions.RequestException as exc:
        return {
            "success": False,
            "text": "",
            "error": f"Something went wrong while fetching the page ({exc.__class__.__name__}).",
            "method": None,
            "title": None,
        }

    if response.status_code == 403:
        return {
            "success": False,
            "text": "",
            "error": "The website blocked the request (403 Forbidden). It may not allow automated access.",
            "method": None,
            "title": None,
        }
    if response.status_code == 404:
        return {
            "success": False,
            "text": "",
            "error": "The page was not found (404). Please double-check the URL.",
            "method": None,
            "title": None,
        }
    if response.status_code >= 400:
        return {
            "success": False,
            "text": "",
            "error": f"The website returned an error (HTTP {response.status_code}).",
            "method": None,
            "title": None,
        }

    html = response.text
    if not html or not html.strip():
        return {
            "success": False,
            "text": "",
            "error": "The page appears to be empty.",
            "method": None,
            "title": None,
        }

    # Extract a page title for context, best-effort
    title = None
    try:
        soup_title = BeautifulSoup(html, "html.parser")
        if soup_title.title and soup_title.title.string:
            title = soup_title.title.string.strip()
    except Exception:
        title = None

    # --- Try trafilatura first ---------------------------------------------
    extracted_text = ""
    method_used = None

    if trafilatura is not None:
        try:
            extracted_text = trafilatura.extract(
                html, include_comments=False, include_tables=True, favor_recall=True
            ) or ""
        except Exception:
            extracted_text = ""

    extracted_text = _clean_whitespace(extracted_text)

    if len(extracted_text) >= MIN_TRAFILATURA_CHARS:
        method_used = "trafilatura"
    else:
        # --- Fallback to BeautifulSoup (tag-based) --------------------------
        bs4_text = _bs4_fallback_extract(html)
        if len(bs4_text) > len(extracted_text):
            extracted_text = bs4_text
            method_used = "beautifulsoup"
        elif extracted_text:
            method_used = "trafilatura"

    # --- Last-resort: grab all visible body text if still too thin -----------
    if len(extracted_text) < MIN_TRAFILATURA_CHARS:
        last_resort_text = _bs4_last_resort_extract(html)
        if len(last_resort_text) > len(extracted_text):
            extracted_text = last_resort_text
            method_used = "beautifulsoup (broad)"

    # --- Always append structured JSON-LD data if present ---------------------
    # This captures dates/facts (e.g. schema.org Event startDate/endDate) that
    # visible-text extraction methods routinely miss, regardless of which
    # method above produced the main body text.
    structured_data = _extract_json_ld_data(html)
    if structured_data:
        extracted_text = (extracted_text + "\n\n" + structured_data).strip() if extracted_text else structured_data

    # --- If content is still thin, try pulling embedded app-state JSON --------
    # (Next.js __NEXT_DATA__ and similar). This is free (no extra request).
    if len(extracted_text.split()) < LOW_CONTENT_WORD_THRESHOLD:
        embedded_state_text = _extract_embedded_state_text(html)
        if embedded_state_text:
            extracted_text = (extracted_text + "\n\n" + embedded_state_text).strip()
            if method_used is None:
                method_used = "embedded app state"

    # --- Still thin? This page is likely JS-rendered client-side. Try a real
    # headless-browser render if Playwright is available (opt-in, local only).
    if len(extracted_text.split()) < LOW_CONTENT_WORD_THRESHOLD and PLAYWRIGHT_AVAILABLE:
        rendered_html = _render_with_playwright(url)
        if rendered_html and rendered_html != html:
            rendered_text = ""
            if trafilatura is not None:
                try:
                    rendered_text = trafilatura.extract(
                        rendered_html, include_comments=False, include_tables=True, favor_recall=True
                    ) or ""
                except Exception:
                    rendered_text = ""
            rendered_text = _clean_whitespace(rendered_text)
            if len(rendered_text) < MIN_TRAFILATURA_CHARS:
                rendered_text = _bs4_last_resort_extract(rendered_html)
            rendered_structured = _extract_json_ld_data(rendered_html)
            if rendered_structured:
                rendered_text = (rendered_text + "\n\n" + rendered_structured).strip()

            if len(rendered_text.split()) > len(extracted_text.split()):
                extracted_text = rendered_text
                method_used = "playwright (rendered)"

    if not extracted_text or not extracted_text.strip():
        # Distinguish "page is basically empty HTML" (JS-rendered SPA) from
        # "we got HTML but truly found no readable text" for a clearer message.
        raw_body_len = len(BeautifulSoup(html, "html.parser").get_text(strip=True))
        if raw_body_len < 50:
            error_msg = (
                "This page appears to render its content with JavaScript — the "
                "raw HTML has almost no text. This scraper (requests + "
                "BeautifulSoup/trafilatura) can't execute JavaScript, so "
                "JS-heavy single-page apps won't work. Try a different page, "
                "or use a site that renders content server-side."
            )
        else:
            error_msg = (
                "Couldn't extract readable article text from this page, even "
                "though it has content. It may use an unusual layout, be "
                "mostly navigation/ads, or be blocked from scraping."
            )
        return {
            "success": False,
            "text": "",
            "error": error_msg,
            "method": None,
            "title": title,
        }

    return {
        "success": True,
        "text": extracted_text,
        "error": None,
        "method": method_used,
        "title": title,
    }


# ---------------------------------------------------------------------------
# OCR / visual extraction helpers
# ---------------------------------------------------------------------------

def _ocr_image_bytes(image_bytes: bytes, lang: str = "eng") -> str:
    """OCR an image in memory. Returns empty text when OCR is unavailable/fails."""
    if pytesseract is None or Image is None:
        return ""
    try:
        image = Image.open(io.BytesIO(image_bytes))
        # A moderate upscale often improves OCR on screenshots/scans.
        if max(image.size) < 1800:
            scale = 1800 / max(image.size)
            image = image.resize(
                (max(1, int(image.width * scale)), max(1, int(image.height * scale)))
            )
        text = pytesseract.image_to_string(image, lang=lang)
        return _clean_whitespace(text or "")
    except Exception:
        return ""


def _ocr_pdf_bytes(file_bytes: bytes, dpi: int = 180) -> str:
    """Render PDF pages and OCR them when normal PDF text extraction is empty."""
    if fitz is None or pytesseract is None or Image is None:
        return ""

    chunks = []
    try:
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        for page_no, page in enumerate(doc, start=1):
            pix = page.get_pixmap(dpi=dpi, alpha=False)
            image_bytes = pix.tobytes("png")
            text = _ocr_image_bytes(image_bytes)
            if text:
                chunks.append(f"Page {page_no}\n{text}")
        doc.close()
    except Exception:
        return ""
    return _clean_whitespace("\n\n".join(chunks))


def _ocr_embedded_office_images(file_bytes: bytes, extension: str) -> str:
    """OCR embedded images from DOCX/PPTX containers."""
    if pytesseract is None or Image is None:
        return ""
    if extension not in ("docx", "pptx"):
        return ""

    chunks = []
    try:
        with zipfile.ZipFile(io.BytesIO(file_bytes)) as zf:
            media_files = [
                name for name in zf.namelist()
                if name.startswith("word/media/") or name.startswith("ppt/media/")
            ]
            for name in media_files:
                text = _ocr_image_bytes(zf.read(name))
                if text:
                    chunks.append(f"Embedded image: {name.split('/')[-1]}\n{text}")
    except Exception:
        return ""
    return _clean_whitespace("\n\n".join(chunks))


def _ocr_dependency_message() -> str:
    return (
        "OCR is not available. Install the Python packages "
        "'pytesseract', 'Pillow', and 'PyMuPDF', and install the "
        "Tesseract OCR application on Windows. Then restart Streamlit."
    )


# ---------------------------------------------------------------------------
# Public: PDF scraping
# ---------------------------------------------------------------------------

def scrape_pdf(url: str) -> dict:
    """
    Download a PDF and extract its text using pdfplumber.

    Returns the same shaped dict as scrape_website().
    """
    url = url.strip()

    if not is_valid_url(url):
        return {
            "success": False,
            "text": "",
            "error": "That doesn't look like a valid URL.",
            "method": None,
            "title": None,
        }

    if pdfplumber is None:
        return {
            "success": False,
            "text": "",
            "error": "PDF support isn't available (pdfplumber not installed).",
            "method": None,
            "title": None,
        }

    try:
        response = requests.get(url, headers=DEFAULT_HEADERS, timeout=REQUEST_TIMEOUT)
    except requests.exceptions.Timeout:
        return {
            "success": False,
            "text": "",
            "error": "The PDF took too long to download (timeout).",
            "method": None,
            "title": None,
        }
    except requests.exceptions.RequestException as exc:
        return {
            "success": False,
            "text": "",
            "error": f"Could not download the PDF ({exc.__class__.__name__}).",
            "method": None,
            "title": None,
        }

    if response.status_code >= 400:
        return {
            "success": False,
            "text": "",
            "error": f"The server returned an error while fetching the PDF (HTTP {response.status_code}).",
            "method": None,
            "title": None,
        }

    try:
        pages_text = []
        with pdfplumber.open(io.BytesIO(response.content)) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text() or ""
                if page_text.strip():
                    pages_text.append(page_text)
        full_text = _clean_whitespace("\n\n".join(pages_text))
    except Exception:
        return {
            "success": False,
            "text": "",
            "error": (
                "Couldn't read this PDF. It may be scanned/image-based (no "
                "extractable text) or corrupted."
            ),
            "method": None,
            "title": None,
        }

    if not full_text:
        return {
            "success": False,
            "text": "",
            "error": (
                "No extractable text was found in this PDF. It may be a scanned "
                "document (image-only), which requires OCR support."
            ),
            "method": None,
            "title": None,
        }

    title = url.split("/")[-1]
    return {
        "success": True,
        "text": full_text,
        "error": None,
        "method": "pdfplumber",
        "title": title,
    }


# ---------------------------------------------------------------------------
# Public: uploaded-file scraping (PDF / DOCX / TXT)
# ---------------------------------------------------------------------------

def scrape_uploaded_pdf(file_bytes: bytes, filename: str) -> dict:
    """
    Extract text from an uploaded PDF.
    Strategy:
      1. pdfplumber for native/selectable PDF text.
      2. PyMuPDF + Tesseract OCR for scanned/image-only PDFs.
    """
    native_text = ""
    native_error = None

    if pdfplumber is not None:
        try:
            pages_text = []
            with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text() or ""
                    if page_text.strip():
                        pages_text.append(page_text)
            native_text = _clean_whitespace("\n\n".join(pages_text))
        except Exception as exc:
            native_error = exc.__class__.__name__

    if native_text:
        return {
            "success": True,
            "text": native_text,
            "error": None,
            "method": "pdfplumber (upload)",
            "title": filename,
            "ocr_used": False,
        }

    # Native text was empty: automatically try OCR.
    ocr_text = _ocr_pdf_bytes(file_bytes)
    if ocr_text:
        return {
            "success": True,
            "text": ocr_text,
            "error": None,
            "method": "PyMuPDF + Tesseract OCR (scanned PDF)",
            "title": filename,
            "ocr_used": True,
        }

    if pdfplumber is None:
        error = "PDF text extraction is unavailable and OCR could not be used. " + _ocr_dependency_message()
    elif native_error:
        error = (
            "Couldn't read the PDF with the normal extractor, and OCR also failed. "
            + _ocr_dependency_message()
        )
    else:
        error = (
            "No selectable text was found. This appears to be a scanned/image-only PDF, "
            "but OCR could not extract text. " + _ocr_dependency_message()
        )

    return {
        "success": False,
        "text": "",
        "error": error,
        "method": None,
        "title": filename,
        "ocr_used": False,
    }


def scrape_uploaded_docx(file_bytes: bytes, filename: str) -> dict:
    """
    Extract text from an in-memory .docx file using python-docx.
    Pulls paragraph text and table cell text. Returns the same shaped
    dict as scrape().
    """
    try:
        import docx  # python-docx
    except ImportError:
        return {
            "success": False,
            "text": "",
            "error": "DOCX support isn't available (python-docx not installed).",
            "method": None,
            "title": filename,
        }

    try:
        document = docx.Document(io.BytesIO(file_bytes))
        chunks = [p.text for p in document.paragraphs if p.text.strip()]
        for table in document.tables:
            for row in table.rows:
                for cell in row.cells:
                    if cell.text.strip():
                        chunks.append(cell.text.strip())
        full_text = _clean_whitespace("\n".join(chunks))
    except Exception:
        return {
            "success": False,
            "text": "",
            "error": "Couldn't read this Word document. It may be corrupted or in an unsupported .doc (not .docx) format.",
            "method": None,
            "title": filename,
        }

    if not full_text:
        return {
            "success": False,
            "text": "",
            "error": "No readable text was found in this document.",
            "method": None,
            "title": filename,
        }

    # Also inspect embedded screenshots/scanned pages when present.
    image_text = _ocr_embedded_office_images(file_bytes, "docx")
    if image_text:
        full_text = _clean_whitespace(full_text + "\n\n" + image_text)

    return {
        "success": True,
        "text": full_text,
        "error": None,
        "method": "python-docx + OCR for embedded images (upload)" if image_text else "python-docx (upload)",
        "title": filename,
        "ocr_used": bool(image_text),
    }


def scrape_uploaded_txt(file_bytes: bytes, filename: str) -> dict:
    """Extract text from an in-memory plain-text (.txt/.md) file."""
    try:
        # Try UTF-8 first, fall back to latin-1 so we never crash on odd encodings.
        try:
            full_text = file_bytes.decode("utf-8")
        except UnicodeDecodeError:
            full_text = file_bytes.decode("latin-1")
        full_text = _clean_whitespace(full_text)
    except Exception:
        return {
            "success": False,
            "text": "",
            "error": "Couldn't read this text file.",
            "method": None,
            "title": filename,
        }

    if not full_text:
        return {
            "success": False,
            "text": "",
            "error": "This file appears to be empty.",
            "method": None,
            "title": filename,
        }

    return {
        "success": True,
        "text": full_text,
        "error": None,
        "method": "plain text (upload)",
        "title": filename,
    }



def scrape_uploaded_excel(file_bytes: bytes, filename: str) -> dict:
    """Extract readable text from .xlsx/.xls Excel workbooks."""
    if pd is None:
        return {
            "success": False,
            "text": "",
            "error": "Excel support isn't available (pandas not installed).",
            "method": None,
            "title": filename,
        }

    try:
        ext = filename.lower().rsplit(".", 1)[-1]
        engine = "openpyxl" if ext == "xlsx" else None
        sheets = pd.read_excel(io.BytesIO(file_bytes), sheet_name=None, engine=engine)
        chunks = []

        for sheet_name, df in sheets.items():
            chunks.append(f"Sheet: {sheet_name}")
            if df is not None and not df.empty:
                # Keep headers and rows readable for the LLM.
                chunks.append(df.fillna("").to_csv(index=False))
            else:
                chunks.append("(empty sheet)")

        full_text = _clean_whitespace("\n".join(chunks))
    except Exception as exc:
        return {
            "success": False,
            "text": "",
            "error": (
                "Couldn't read this Excel file. Make sure it is a valid "
                f".xlsx/.xls workbook. ({exc.__class__.__name__})"
            ),
            "method": None,
            "title": filename,
        }

    if not full_text:
        return {
            "success": False,
            "text": "",
            "error": "This Excel file appears to be empty.",
            "method": None,
            "title": filename,
        }

    return {
        "success": True,
        "text": full_text,
        "error": None,
        "method": "pandas (Excel upload)",
        "title": filename,
    }


def scrape_uploaded_csv(file_bytes: bytes, filename: str) -> dict:
    """Extract readable text from a CSV upload."""
    if pd is None:
        return {
            "success": False,
            "text": "",
            "error": "CSV support isn't available (pandas not installed).",
            "method": None,
            "title": filename,
        }

    try:
        df = pd.read_csv(io.BytesIO(file_bytes))
        full_text = _clean_whitespace(df.fillna("").to_csv(index=False))
    except Exception as exc:
        return {
            "success": False,
            "text": "",
            "error": f"Couldn't read this CSV file ({exc.__class__.__name__}).",
            "method": None,
            "title": filename,
        }

    if not full_text:
        return {
            "success": False,
            "text": "",
            "error": "This CSV file appears to be empty.",
            "method": None,
            "title": filename,
        }

    return {
        "success": True,
        "text": full_text,
        "error": None,
        "method": "pandas (CSV upload)",
        "title": filename,
    }


def scrape_uploaded_pptx(file_bytes: bytes, filename: str) -> dict:
    """Extract text from PowerPoint .pptx slides."""
    try:
        from pptx import Presentation
    except ImportError:
        return {
            "success": False,
            "text": "",
            "error": "PPTX support isn't available (python-pptx not installed).",
            "method": None,
            "title": filename,
        }

    try:
        prs = Presentation(io.BytesIO(file_bytes))
        chunks = []
        for slide_no, slide in enumerate(prs.slides, start=1):
            slide_parts = []
            for shape in slide.shapes:
                if hasattr(shape, "text") and shape.text.strip():
                    slide_parts.append(shape.text.strip())
            if slide_parts:
                chunks.append(f"Slide {slide_no}\n" + "\n".join(slide_parts))

        full_text = _clean_whitespace("\n\n".join(chunks))
    except Exception as exc:
        return {
            "success": False,
            "text": "",
            "error": f"Couldn't read this PowerPoint file ({exc.__class__.__name__}).",
            "method": None,
            "title": filename,
        }

    if not full_text:
        return {
            "success": False,
            "text": "",
            "error": "No readable text was found in this PowerPoint file.",
            "method": None,
            "title": filename,
        }

    image_text = _ocr_embedded_office_images(file_bytes, "pptx")
    if image_text:
        full_text = _clean_whitespace(full_text + "\n\n" + image_text)

    return {
        "success": True,
        "text": full_text,
        "error": None,
        "method": "python-pptx + OCR for embedded images (upload)" if image_text else "python-pptx (upload)",
        "title": filename,
        "ocr_used": bool(image_text),
    }


def scrape_uploaded_image(file_bytes: bytes, filename: str) -> dict:
    """OCR an uploaded image (PNG/JPG/JPEG/WEBP/BMP/TIFF)."""
    text = _ocr_image_bytes(file_bytes)
    if not text:
        return {
            "success": False,
            "text": "",
            "error": (
                "No readable text was found in this image. "
                + _ocr_dependency_message()
            ),
            "method": None,
            "title": filename,
        }

    return {
        "success": True,
        "text": text,
        "error": None,
        "method": "Tesseract OCR (image upload)",
        "title": filename,
        "ocr_used": True,
    }


def scrape_uploaded_file(file_bytes: bytes, filename: str) -> dict:
    """
    Unified entry point for uploaded files: routes to the right extractor
    based on file extension, then attaches language/word-count metadata
    just like scrape() does for URLs.
    """
    ext = filename.lower().rsplit(".", 1)[-1] if "." in filename else ""

    if ext == "pdf":
        result = scrape_uploaded_pdf(file_bytes, filename)
    elif ext == "docx":
        result = scrape_uploaded_docx(file_bytes, filename)
    elif ext in ("txt", "md"):
        result = scrape_uploaded_txt(file_bytes, filename)
    elif ext in ("xlsx", "xls"):
        result = scrape_uploaded_excel(file_bytes, filename)
    elif ext == "csv":
        result = scrape_uploaded_csv(file_bytes, filename)
    elif ext == "pptx":
        result = scrape_uploaded_pptx(file_bytes, filename)
    elif ext in ("png", "jpg", "jpeg", "webp", "bmp", "tif", "tiff"):
        result = scrape_uploaded_image(file_bytes, filename)
    elif ext == "doc":
        result = {
            "success": False,
            "text": "",
            "error": (
                "Old-style .doc files aren't supported — please save/export it as "
                ".docx (Word) or .pdf and upload that instead."
            ),
            "method": None,
            "title": filename,
        }
    else:
        result = {
            "success": False,
            "text": "",
            "error": (
                f"Unsupported file type '.{ext}'. Please upload PDF, DOCX, TXT, "
                "MD, XLSX, XLS, CSV, PPTX, PNG, JPG, JPEG, WEBP, BMP, or TIFF."
            ),
            "method": None,
            "title": filename,
        }

    if result["success"]:
        result["language"] = detect_language(result["text"])
        result["word_count"] = len(result["text"].split())
        result["char_count"] = len(result["text"])
        result["is_low_content"] = result["word_count"] < LOW_CONTENT_WORD_THRESHOLD
    else:
        result["language"] = None
        result["word_count"] = 0
        result["char_count"] = 0
        result["is_low_content"] = False

    return result


# ---------------------------------------------------------------------------
# Public: unified entry point
# ---------------------------------------------------------------------------

def scrape(url: str) -> dict:
    """
    Unified entry point: detects whether the URL is a PDF or HTML page
    and routes to the correct scraper.
    """
    url = url.strip()
    if not is_valid_url(url):
        return {
            "success": False,
            "text": "",
            "error": "That doesn't look like a valid URL. Please include https:// or http://",
            "method": None,
            "title": None,
        }

    if is_pdf_url(url):
        result = scrape_pdf(url)
    else:
        result = scrape_website(url)

    if result["success"]:
        result["language"] = detect_language(result["text"])
        result["word_count"] = len(result["text"].split())
        result["char_count"] = len(result["text"])
        result["is_low_content"] = result["word_count"] < LOW_CONTENT_WORD_THRESHOLD
    else:
        result["language"] = None
        result["word_count"] = 0
        result["char_count"] = 0
        result["is_low_content"] = False

    return result