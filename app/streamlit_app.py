import streamlit as st
import os
import sys
from pathlib import Path
from datetime import datetime
import time

# Make python-dotenv optional (no-op if unavailable)
try:
    from dotenv import load_dotenv  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    def load_dotenv():  # type: ignore
        return False

sys.path.append(str(Path(__file__).parent.parent))

from src.agents.search_agent import SearchAgent
from src.agents.synthesizer_agent import SynthesizerAgent

# --- Page Config ---
st.set_page_config(
    page_title="Agentic Research Assistant",
    page_icon="🔬",
    layout="centered"
)

# --- Light Mode Custom CSS ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    .stApp {
        background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 50%, #f1f5f9 100%);
        color: #1e293b;
        font-family: 'Inter', sans-serif;
    }
    
    /* Hide Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .stDeployButton {visibility: hidden;}
    
    /* Header Container */
    .header-container {
        background: rgba(255, 255, 255, 0.9);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(0, 0, 0, 0.05);
        border-radius: 16px;
        padding: 1rem 2rem;
        margin: -1rem 0 2rem 0;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
        display: flex;
        align-items: center;
        justify-content: space-between;
    }
    
    .header-logo {
        font-size: 1.3rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    .header-nav {
        display: flex;
        gap: 2rem;
        align-items: center;
        flex-grow: 1;
        justify-content: center;
    }
    
    .nav-item {
        color: #64748b;
        font-weight: 500;
        font-size: 0.9rem;
        padding: 0.5rem 1rem;
        border-radius: 8px;
        transition: all 0.3s ease;
        cursor: pointer;
        text-decoration: none;
    }
    
    .nav-item:hover {
        color: #667eea;
        background: rgba(102, 126, 234, 0.1);
    }
    
    .nav-item.active {
        color: #667eea;
        background: rgba(102, 126, 234, 0.15);
        font-weight: 600;
    }
    
    .status-badge {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        border: 1px solid transparent;
    }
    .status-badge.ok { background: rgba(34, 197, 94, 0.1); color: #22c55e; border-color: rgba(34, 197, 94, 0.2); }
    .status-badge.warn { background: rgba(245, 158, 11, 0.1); color: #ca8a04; border-color: rgba(245, 158, 11, 0.2); }
    .status-badge.error { background: rgba(239, 68, 68, 0.1); color: #dc2626; border-color: rgba(239, 68, 68, 0.2); }
    
    .status-dot { width: 8px; height: 8px; border-radius: 50%; animation: pulse 2s infinite; }
    .status-dot.ok { background: #22c55e; }
    .status-dot.warn { background: #f59e0b; }
    .status-dot.error { background: #ef4444; }
    
    /* Main Content */
    .main-header {
        text-align: center;
        font-size: 2.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin: 1rem 0 0.5rem 0;
        letter-spacing: -0.02em;
    }

    .tagline {
        text-align: center;
        color: #64748b;
        font-size: 1.1rem;
        margin-bottom: 2rem;
        font-weight: 400;
    }

    /* Input Styling */
    .stTextInput > div > div > input {
        background: rgba(255, 255, 255, 0.9) !important;
        border: 1px solid #e2e8f0 !important;
        border-radius: 12px !important;
        color: #1e293b !important;
        font-size: 1rem !important;
        padding: 0.8rem 1rem !important;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.04) !important;
        transition: all 0.3s ease !important;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #667eea !important;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1) !important;
    }
    
    .stTextInput > div > div > input::placeholder {
        color: #94a3b8 !important;
    }

    /* Button Styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        border: none !important;
        border-radius: 12px !important;
        color: white !important;
        font-size: 1.1rem !important;
        font-weight: 600 !important;
        padding: 0.75rem 2rem !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 20px rgba(102, 126, 234, 0.3) !important;
        min-width: 300px !important;
    }

    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 30px rgba(102, 126, 234, 0.4) !important;
    }
    
    .stButton > button:active {
        transform: translateY(0px) !important;
    }

    /* Response Card */
    .response-card {
        background: rgba(255, 255, 255, 0.9);
        border: 1px solid rgba(0, 0, 0, 0.05);
        border-radius: 16px;
        padding: 1.5rem;
        margin-top: 1.5rem;
        backdrop-filter: blur(10px);
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
        animation: fadeIn 0.8s ease;
    }

    /* Progress Bar */
    .stProgress > div > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%) !important;
    }
    
    .stProgress > div {
        background-color: rgba(0, 0, 0, 0.1) !important;
    }
    
    /* Success/Error Messages */
    .stSuccess {
        background-color: rgba(34, 197, 94, 0.1) !important;
        color: #16a34a !important;
        border: 1px solid rgba(34, 197, 94, 0.2) !important;
    }
    
    .stError {
        background-color: rgba(239, 68, 68, 0.1) !important;
        color: #dc2626 !important;
        border: 1px solid rgba(239, 68, 68, 0.2) !important;
    }

    /* Download Button */
    .stDownloadButton > button {
        background: rgba(34, 197, 94, 0.1) !important;
        border: 1px solid rgba(34, 197, 94, 0.3) !important;
        border-radius: 12px !important;
        color: #16a34a !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
    }
    
    .stDownloadButton > button:hover {
        background: rgba(34, 197, 94, 0.2) !important;
        transform: translateY(-1px) !important;
    }

    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }

    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }

    /* Mobile Responsive */
    @media (max-width: 768px) {
        .header-container {
            flex-direction: column;
            gap: 1rem;
            padding: 1rem;
        }
        
        .header-nav {
            gap: 1rem;
            justify-content: center;
        }
        
        .nav-item {
            font-size: 0.8rem;
            padding: 0.4rem 0.8rem;
        }
        
        .main-header {
            font-size: 2rem;
        }
        
        .stButton > button {
            min-width: 250px !important;
        }
    }
</style>
""", unsafe_allow_html=True)

# --- Fixed Header using HTML ---
# Build dynamic status badge
_processing = st.session_state.get("processing", False)
if api_key and not _processing:
    _status_cls, _status_text = "ok", "Online"
elif _processing:
    _status_cls, _status_text = "warn", "Processing..."
else:
    _status_cls, _status_text = "error", "No API key"

_header_html = f"""
<div class="header-container">
    <div class="header-logo">🔬 Research Assistant</div>
    <div class="header-nav">
        <div class="nav-item active">Dashboard</div>
        <div class="nav-item">History</div>
        <div class="nav-item">Settings</div>
        <div class="nav-item">Help</div>
    </div>
    <div class="status-badge {_status_cls}">
        <div class="status-dot {_status_cls}"></div>
        {_status_text}
    </div>
</div>
"""
st.markdown(_header_html, unsafe_allow_html=True)

# --- Main Content ---
st.markdown('<h1 class="main-header">🔬 Agentic Research Assistant</h1>', unsafe_allow_html=True)
st.markdown('<p class="tagline">Your Intelligent Literature Companion</p>', unsafe_allow_html=True)

# --- Load Env and Initialize ---
load_dotenv()
# Prefer GEMINI_API_KEY, fallback to API_KEY and Streamlit secrets
_secrets_key = None
try:
    _secrets_key = st.secrets["GEMINI_API_KEY"]
except Exception:
    try:
        _secrets_key = st.secrets["API_KEY"]
    except Exception:
        _secrets_key = None

api_key = os.getenv("GEMINI_API_KEY") or os.getenv("API_KEY") or _secrets_key

if "review" not in st.session_state:
    st.session_state.review = None
if "processing" not in st.session_state:
    st.session_state.processing = False

# --- Input Section (centered) ---
_in_col1, _in_col2, _in_col3 = st.columns([1, 2, 1])
with _in_col2:
    query = st.text_input(
        "Enter your research query",
        placeholder="e.g. Deep learning for time series forecasting",
        key="query",
    )
    if not api_key:
        st.warning("No API key configured. Set `GEMINI_API_KEY` in .env or Streamlit secrets.")
    else:
        st.caption("API key detected.")

# --- Centered Button ---
_btn_col1, _btn_col2, _btn_col3 = st.columns([1, 2, 1])
with _btn_col2:
    generate = st.button("🚀 Generate Literature Review", disabled=not api_key or not query)
    if st.session_state.get("processing"):
        st.info("⏳ Generating literature review... This may take a minute.")

# --- Generation Logic ---
if generate and query:
    st.session_state.processing = True
    st.session_state.review = None

    try:
        progress = st.progress(0)
        status = st.empty()
        status.text("🔧 Initializing...")

        search = SearchAgent(pdf_dir="pdfs")
        # Lazy import to avoid hard dependency at app startup
        try:
            from src.agents.summarizer_agent import SummarizerAgent  # noqa: WPS433
        except Exception as import_err:
            st.error(
                "Missing dependency for summarization. Install 'google-generativeai' and 'sentence-transformers'.\n"
                f"Import error: {import_err}"
            )
            raise
        summarizer = SummarizerAgent(api_key=api_key)
        synthesizer = SynthesizerAgent()

        progress.progress(10)
        status.text(f"🔍 Searching arXiv for '{query}'")

        papers = search.search_arxiv(query, max_results=5)
        summaries = []

        for i, paper in enumerate(papers):
            status.text(f"📄 Processing: {paper['title'][:60]}...")
            progress.progress(20 + int((i / len(papers)) * 50))
            summary = summarizer._summarize_text(paper.get("summary", ""), max_output_tokens=300)
            summaries.append(summary)
            time.sleep(0.5)

        progress.progress(80)
        status.text("🧠 Synthesizing literature review...")

        review = synthesizer.synthesize(papers, summaries)
        st.session_state.review = review

        progress.progress(100)
        status.text("✅ Complete!")
        st.success("Literature Review Generated Successfully!")

    except Exception as e:
        st.error(f"❌ Error: {str(e)}")

    finally:
        st.session_state.processing = False

# --- Output Section ---
if st.session_state.review:
    st.markdown('<div class="response-card">', unsafe_allow_html=True)
    st.markdown("### 📝 Generated Literature Review")
    st.markdown(st.session_state.review)
    dl_col, clr_col = st.columns([3, 1])
    with dl_col:
        st.download_button(
            "📥 Download Markdown",
            st.session_state.review,
            file_name=f"review_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
            mime="text/markdown"
        )
    with clr_col:
        if st.button("🧹 Clear"):
            st.session_state.review = None
            st.session_state.processing = False
            try:
                st.session_state.query = ""
            except Exception:
                pass
            st.experimental_rerun()
    st.markdown("</div>", unsafe_allow_html=True)
