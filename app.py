"""
Research Paper Summarizer - Streamlit Application
A professional AI agent for efficiently summarizing research papers using Groq LLM.
"""

import os
import sys
import secrets
from pathlib import Path
from dotenv import load_dotenv
import streamlit as st
from streamlit import runtime as st_runtime
from agent import (
    ResearchPaperSummarizer,
    PaperAnalyzer,
    SummaryType,
    PaperMetadata,
    QuotaExceededError,
)
import time
from datetime import datetime
import json

load_dotenv()

TOKEN_STORE_PATH = Path(__file__).with_name("anonymous_usage.json")
ANON_TOKEN_LIMIT = int(os.getenv("ANON_USER_TOKEN_LIMIT", "7000"))


# When launched with `python app.py` or `uv run app.py`, hand off to
# Streamlit's app runner so session state and UI context work correctly.
if __name__ == "__main__" and not st_runtime.exists():
    from streamlit.web import cli as stcli

    sys.argv = ["streamlit", "run", str(Path(__file__).resolve()), *sys.argv[1:]]
    raise SystemExit(stcli.main())


# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="Research Paper Summarizer",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for professional styling
st.markdown("""
<style>
    /* Main container styling */
    .main {
        padding: 2rem;
    }
    
    /* Header styling */
    .header-title {
        font-size: 2.5rem;
        font-weight: 700;
        background: #16324f;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0.5rem;
    }
    
    .header-subtitle {
        font-size: 1.1rem;
        color: #666;
        margin-bottom: 2rem;
    }
    
    /* Card styling */
    .summary-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        border-left: 4px solid #667eea;
    }
    
    .metric-card {
        background: #f8f9fa;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
        border: 1px solid #e0e0e0;
    }
    
    /* Button styling */
    .stButton > button {
        background: #16324f;
        color: #f8fbff;
        border: 1px solid #254c77;
        border-radius: 999px;
        padding: 0.8rem 1.4rem;
        font-weight: 700;
        letter-spacing: 0.02em;
        box-shadow: 0 10px 22px rgba(22, 50, 79, 0.18);
        transition: transform 0.2s ease, box-shadow 0.2s ease, background 0.2s ease;
    }
    
    .stButton > button:hover {
        background: #1f5f8b;
        border-color: #1f5f8b;
        transform: translateY(-1px);
        box-shadow: 0 14px 28px rgba(31, 95, 139, 0.25);
    }

    .stButton > button:focus {
        outline: none;
        box-shadow: 0 0 0 0.2rem rgba(31, 95, 139, 0.2);
    }
    
    /* Sidebar styling */
    .sidebar .sidebar-content {
        padding: 2rem 1rem;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] button {
        font-weight: 600;
        border-radius: 8px 8px 0 0;
    }
    
    /* Info boxes */
    .info-box {
        background: #e3f2fd;
        border-left: 4px solid #2196f3;
        padding: 1rem;
        border-radius: 4px;
        margin: 1rem 0;
    }
    
    .success-box {
        background: #e8f5e9;
        border-left: 4px solid #4caf50;
        padding: 1rem;
        border-radius: 4px;
        margin: 1rem 0;
    }
    
    .warning-box {
        background: #fff3e0;
        border-left: 4px solid #ff9800;
        padding: 1rem;
        border-radius: 4px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================

def load_token_store() -> dict:
    """Load persisted token usage per anonymous user."""
    if not TOKEN_STORE_PATH.exists():
        return {}

    try:
        return json.loads(TOKEN_STORE_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def save_token_store(store: dict) -> None:
    """Persist token usage per anonymous user."""
    TOKEN_STORE_PATH.write_text(json.dumps(store, indent=2), encoding="utf-8")


def get_or_create_anonymous_key() -> str:
    """Get a stable anonymous key for the current visitor."""
    if "anonymous_key" in st.session_state and st.session_state.anonymous_key:
        return st.session_state.anonymous_key

    query_key = st.query_params.get("anon_key")
    if isinstance(query_key, list):
        query_key = query_key[0] if query_key else ""

    anonymous_key = query_key or f"anon_{secrets.token_hex(8)}"
    st.session_state.anonymous_key = anonymous_key

    if not query_key:
        st.query_params["anon_key"] = anonymous_key

    return anonymous_key


def get_user_token_usage(anonymous_key: str) -> int:
    """Return total recorded usage for an anonymous user."""
    store = load_token_store()
    return int(store.get(anonymous_key, {}).get("used_tokens", 0))


def get_remaining_tokens(anonymous_key: str) -> int:
    """Return remaining token budget for an anonymous user."""
    return max(0, ANON_TOKEN_LIMIT - get_user_token_usage(anonymous_key))


def ensure_token_budget(anonymous_key: str, estimated_tokens: int) -> None:
    """Block requests that would exceed the anonymous quota."""
    remaining_tokens = get_remaining_tokens(anonymous_key)
    if estimated_tokens > remaining_tokens:
        raise QuotaExceededError(
            f"Anonymous quota exceeded. This request may use about {estimated_tokens:,} tokens, "
            f"but only {remaining_tokens:,} remain for this user key."
        )


def record_token_usage(anonymous_key: str, used_tokens: int) -> None:
    """Persist actual usage after a successful Groq response."""
    if used_tokens <= 0:
        return

    store = load_token_store()
    now = datetime.now().isoformat(timespec="seconds")
    user_record = store.get(
        anonymous_key,
        {
            "used_tokens": 0,
            "created_at": now,
        },
    )
    user_record["used_tokens"] = int(user_record.get("used_tokens", 0)) + int(used_tokens)
    user_record["updated_at"] = now
    store[anonymous_key] = user_record
    save_token_store(store)


def build_summarizer(api_key: str | None):
    """Create a summarizer instance when an API key is available."""
    if not api_key:
        return None

    summarizer = ResearchPaperSummarizer(api_key)
    anonymous_key = get_or_create_anonymous_key()
    summarizer.configure_usage_hooks(
        quota_preflight=lambda estimated_tokens: ensure_token_budget(anonymous_key, estimated_tokens),
        usage_recorder=lambda used_tokens: record_token_usage(anonymous_key, used_tokens),
    )
    return summarizer


def initialize_session_state():
    """Initialize Streamlit session state variables."""
    get_or_create_anonymous_key()

    if "api_key" not in st.session_state:
        st.session_state.api_key = os.getenv("GROQ_API_KEY", "")

    if "api_error" not in st.session_state:
        st.session_state.api_error = None

    if "summarizer" not in st.session_state:
        try:
            st.session_state.summarizer = build_summarizer(st.session_state.api_key)
        except Exception as e:
            st.session_state.summarizer = None
            st.session_state.api_error = str(e)
    
    if "paper_metadata" not in st.session_state:
        st.session_state.paper_metadata = None
    
    if "current_summary" not in st.session_state:
        st.session_state.current_summary = None
    
    if "analysis_results" not in st.session_state:
        st.session_state.analysis_results = None
    
    if "processing" not in st.session_state:
        st.session_state.processing = False

    if "active_pdf_id" not in st.session_state:
        st.session_state.active_pdf_id = None
    
    if "history" not in st.session_state:
        st.session_state.history = []


initialize_session_state()


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def save_to_history(paper_title: str, summary_type: str, summary: str):
    """Save analysis to session history."""
    st.session_state.history.append({
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "title": paper_title,
        "type": summary_type,
        "summary": summary[:200] + "..." if len(summary) > 200 else summary,
    })


def export_results(results: dict, filename: str = "paper_analysis.json"):
    """Export analysis results to JSON file."""
    return json.dumps(results, indent=2)


def format_text_for_display(text: str, max_length: int = None) -> str:
    """Format text for display in Streamlit."""
    if max_length and len(text) > max_length:
        return text[:max_length] + "..."
    return text


def reset_current_paper_state():
    """Clear analysis state so each session works with one PDF at a time."""
    st.session_state.current_summary = None
    st.session_state.paper_metadata = None
    st.session_state.analysis_results = None
    st.session_state.processing = False
    st.session_state.full_analysis = False
    if st.session_state.summarizer:
        st.session_state.summarizer.clear_history()


def safe_generate_summary(text: str, summary_type: SummaryType):
    """Generate a summary and show user-friendly quota errors."""
    try:
        return st.session_state.summarizer.summarize(text, summary_type)
    except QuotaExceededError as error:
        st.warning(str(error))
    except Exception as error:
        st.error(f"Error generating summary: {error}")
    return None


def safe_extract_metrics(text: str):
    """Extract metrics and show user-friendly quota errors."""
    try:
        return st.session_state.summarizer.extract_key_metrics(text)
    except QuotaExceededError as error:
        st.warning(str(error))
    except Exception as error:
        st.error(f"Error extracting metrics: {error}")
    return None


def safe_ask_question(question: str):
    """Ask a follow-up question and show user-friendly quota errors."""
    try:
        return st.session_state.summarizer.ask_followup_question(question)
    except QuotaExceededError as error:
        st.warning(str(error))
    except Exception as error:
        st.error(f"Error answering question: {error}")
    return None


# ============================================================================
# HEADER AND NAVIGATION
# ============================================================================

col1, col2 = st.columns([3, 1])

with col1:
    st.markdown('<div class="header-title">📄 Research Paper Summarizer</div>', 
                unsafe_allow_html=True)
    st.markdown(
        '<div class="header-subtitle"> Fast, Efficient, Production-Ready</div>',
        unsafe_allow_html=True
    )



# ============================================================================
# MAIN INTERFACE
# ============================================================================

if not st.session_state.summarizer:
    st.subheader("Connect Groq API")
    api_key_input = st.text_input(
        "Groq API Key",
        value=st.session_state.api_key,
        type="password",
        placeholder="gsk_...",
        help="Paste your Groq API key here. It will stay in this Streamlit session unless you also save it in a .env file.",
    )

    col_save, col_clear = st.columns([1, 1])
    with col_save:
        if st.button("Connect API Key", use_container_width=True):
            try:
                st.session_state.api_key = api_key_input.strip()
                st.session_state.summarizer = build_summarizer(st.session_state.api_key)
                st.session_state.api_error = None
                st.rerun()
            except Exception as e:
                st.session_state.summarizer = None
                st.session_state.api_error = str(e)

    with col_clear:
        if st.button("Clear Key", use_container_width=True):
            st.session_state.api_key = ""
            st.session_state.summarizer = None
            st.session_state.api_error = None
            st.rerun()

    st.error("""
    ### ⚠️ API Key Not Found
    
    Please set your Groq API key as an environment variable:
    ```bash
    export GROQ_API_KEY="your-api-key-here"
    ```
    
    Get your free API key from [console.groq.com](https://console.groq.com)
    """)
    st.info("""
    Windows PowerShell:
    ```powershell
    $env:GROQ_API_KEY="your-api-key-here"
    ```

    Or create a `.env` file in this project folder with:
    ```env
    GROQ_API_KEY=your-api-key-here
    ```
    """)

    if st.session_state.api_error:
        st.warning(f"Connection error: {st.session_state.api_error}")
else:
    # Create tabs for different features
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Summarize", 
        "🔍 Detailed Analysis", 
        "💬 Q&A", 
        "📜 History"
    ])
    
    # ========================================================================
    # TAB 1: SUMMARIZE
    # ========================================================================
    with tab1:
        st.header("Summarize Research Paper")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            uploaded_file = st.file_uploader(
                "Upload a PDF research paper",
                type="pdf",
                help="Select a PDF file to summarize"
            )
        
        with col2:
            summary_type = st.selectbox(
                "Summary Type",
                options=[
                    SummaryType.ABSTRACT,
                    SummaryType.DETAILED,
                    SummaryType.BULLET_POINTS,
                    SummaryType.TECHNICAL,
                    SummaryType.EXECUTIVE,
                ],
                format_func=lambda x: x.value.replace("_", " ").title(),
                help="Choose the type of summary to generate"
            )
        
        if uploaded_file is None and st.session_state.active_pdf_id is not None:
            reset_current_paper_state()
            st.session_state.active_pdf_id = None

        if uploaded_file is not None:
            current_pdf_id = f"{uploaded_file.name}:{uploaded_file.size}"
            if (
                st.session_state.active_pdf_id is not None
                and st.session_state.active_pdf_id != current_pdf_id
            ):
                reset_current_paper_state()

            st.session_state.active_pdf_id = current_pdf_id
            st.markdown('<div class="info-box">', unsafe_allow_html=True)
            st.info(f"📁 File: {uploaded_file.name} | Size: {uploaded_file.size / 1024:.1f} KB")
            st.caption("One PDF is active per user session. Uploading a different PDF replaces the current paper context.")
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Create two columns for buttons
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("🚀 Generate Summary", use_container_width=True):
                    st.session_state.processing = True
            
            with col2:
                if st.button("📊 Full Analysis", use_container_width=True):
                    st.session_state.processing = True
                    st.session_state.full_analysis = True
            
            with col3:
                if st.button("🔄 Clear", use_container_width=True):
                    reset_current_paper_state()
                    st.session_state.active_pdf_id = None
                    st.rerun()
            
            # Process the PDF
            if st.session_state.processing:
                with st.spinner("🔄 Processing PDF..."):
                    try:
                        # Extract text from PDF
                        progress_bar = st.progress(0)
                        
                        progress_bar.progress(20)
                        metadata = st.session_state.summarizer.extract_text_from_pdf(uploaded_file)
                        st.session_state.paper_metadata = metadata
                        
                        progress_bar.progress(40)
                        
                        # Check if full analysis is requested
                        if hasattr(st.session_state, 'full_analysis') and st.session_state.full_analysis:
                            analyzer = PaperAnalyzer(st.session_state.summarizer)
                            progress_bar.progress(60)
                            
                            results = analyzer.comprehensive_analysis(uploaded_file)
                            st.session_state.analysis_results = results
                            
                            progress_bar.progress(100)
                            
                            # Display results
                            st.markdown('<div class="success-box">', unsafe_allow_html=True)
                            st.success("✅ Full analysis completed successfully!")
                            st.markdown('</div>', unsafe_allow_html=True)
                            
                            # Metadata
                            st.subheader("📋 Paper Information")
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Title", results["metadata"]["title"][:30] + "..." if len(str(results["metadata"]["title"])) > 30 else results["metadata"]["title"])
                            with col2:
                                st.metric("Pages", results["metadata"]["pages"])
                            with col3:
                                st.metric("Authors", results["metadata"]["authors"][:20] + "..." if results["metadata"]["authors"] and len(str(results["metadata"]["authors"])) > 20 else results["metadata"]["authors"])
                            
                            # Summaries
                            st.subheader("📝 Summaries")
                            
                            summary_col1, summary_col2 = st.columns(2)
                            
                            with summary_col1:
                                with st.expander("📌 Abstract", expanded=True):
                                    st.write(results["abstract"])
                            
                            with summary_col2:
                                with st.expander("📌 Key Points", expanded=True):
                                    st.write(results["key_points"])
                            
                            with st.expander("📌 Detailed Summary", expanded=False):
                                st.write(results["detailed_summary"])
                            
                            # Metrics
                            if results["metrics"] and "error" not in results["metrics"]:
                                st.subheader("📊 Key Metrics")
                                st.json(results["metrics"])
                            
                            # Export button
                            st.divider()
                            col1, col2 = st.columns(2)
                            with col1:
                                export_data = export_results(results)
                                st.download_button(
                                    label="📥 Download Analysis (JSON)",
                                    data=export_data,
                                    file_name=f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                                    mime="application/json"
                                )
                            
                            with col2:
                                save_to_history(
                                    results["metadata"]["title"],
                                    "Full Analysis",
                                    results["detailed_summary"]
                                )
                                st.success("✅ Saved to history")
                            
                            st.session_state.full_analysis = False
                        
                        else:
                            # Generate single summary
                            progress_bar.progress(60)
                            summary = st.session_state.summarizer.summarize(
                                metadata.extracted_text,
                                summary_type
                            )
                            st.session_state.current_summary = summary
                            
                            progress_bar.progress(100)
                            
                            # Display results
                            st.markdown('<div class="success-box">', unsafe_allow_html=True)
                            st.success(f"✅ {summary_type.value.replace('_', ' ').title()} Summary Generated!")
                            st.markdown('</div>', unsafe_allow_html=True)
                            
                            # Paper info
                            st.subheader("📋 Paper Information")
                            col1, col2 = st.columns(2)
                            with col1:
                                st.write(f"**Pages:** {metadata.total_pages}")
                            with col2:
                                st.write(f"**Characters:** {len(metadata.extracted_text):,}")
                            
                            # Summary
                            st.subheader(f"📝 {summary_type.value.replace('_', ' ').title()} Summary")
                            st.write(summary)
                            
                            # Save and export
                            st.divider()
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                if st.button("💾 Save to History", use_container_width=True):
                                    save_to_history(
                                        metadata.title or "Unknown",
                                        summary_type.value,
                                        summary
                                    )
                                    st.success("✅ Saved to history")
                            
                            with col2:
                                summary_text = f"# {summary_type.value.title()} Summary\n\n{summary}"
                                st.download_button(
                                    label="📥 Download Summary",
                                    data=summary_text,
                                    file_name=f"summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                                    use_container_width=True
                                )
                            
                            with col3:
                                if st.button("🔄 Try Another Type", use_container_width=True):
                                    st.session_state.current_summary = None
                                    st.rerun()
                        
                        st.session_state.processing = False
                    
                    except Exception as e:
                        st.markdown('<div class="warning-box">', unsafe_allow_html=True)
                        st.error(f"❌ Error processing PDF: {str(e)}")
                        st.markdown('</div>', unsafe_allow_html=True)
                        st.session_state.processing = False
    
    # ========================================================================
    # TAB 2: DETAILED ANALYSIS
    # ========================================================================
    with tab2:
        st.header("Detailed Analysis")
        
        if st.session_state.paper_metadata is None:
            st.info("📌 Please upload and process a PDF in the 'Summarize' tab first.")
        else:
            metadata = st.session_state.paper_metadata
            
            st.subheader("📊 Analysis Overview")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Pages", metadata.total_pages)
            with col2:
                st.metric("Text Length", f"{len(metadata.extracted_text):,} chars")
            with col3:
                st.metric("Estimated Reading Time", f"~{len(metadata.extracted_text) // 1000} min")
            
            st.divider()
            
            # Generate different summary types
            st.subheader("🔍 Generate Different Summary Types")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("📌 Abstract Summary", use_container_width=True):
                    with st.spinner("Generating abstract..."):
                        summary = safe_generate_summary(
                            metadata.extracted_text,
                            SummaryType.ABSTRACT
                        )
                        if summary:
                            st.write(summary)
            
            with col2:
                if st.button("🔧 Technical Summary", use_container_width=True):
                    with st.spinner("Generating technical summary..."):
                        summary = safe_generate_summary(
                            metadata.extracted_text,
                            SummaryType.TECHNICAL
                        )
                        if summary:
                            st.write(summary)
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("📋 Bullet Points", use_container_width=True):
                    with st.spinner("Generating bullet points..."):
                        summary = safe_generate_summary(
                            metadata.extracted_text,
                            SummaryType.BULLET_POINTS
                        )
                        if summary:
                            st.write(summary)
            
            with col2:
                if st.button("👥 Executive Summary", use_container_width=True):
                    with st.spinner("Generating executive summary..."):
                        summary = safe_generate_summary(
                            metadata.extracted_text,
                            SummaryType.EXECUTIVE
                        )
                        if summary:
                            st.write(summary)
            
            st.divider()
            
            # Extract metrics
            st.subheader("📊 Key Metrics")
            if st.button("🔍 Extract Metrics", use_container_width=True):
                with st.spinner("Extracting metrics..."):
                    metrics = safe_extract_metrics(metadata.extracted_text)
                    if metrics is not None:
                        st.json(metrics)
    
    # ========================================================================
    # TAB 3: Q&A
    # ========================================================================
    with tab3:
        st.header("Ask Questions About the Paper")
        
        if st.session_state.paper_metadata is None:
            st.info("📌 Please upload and process a PDF in the 'Summarize' tab first.")
        else:
            st.info(
                "💡 Ask follow-up questions about the research paper. "
                "The AI will answer based on the paper content."
            )
            
            # Question input
            question = st.text_area(
                "Your Question:",
                placeholder="Type a question for the AI agent, for example: What problem does this paper solve, what method was used, or what are the main limitations?",
                height=100
            )
            
            col1, col2 = st.columns([3, 1])
            
            with col1:
                ask_button = st.button("🤔 Ask Question", use_container_width=True)
            
            with col2:
                clear_button = st.button("🔄 Clear History", use_container_width=True)
            
            if clear_button:
                st.session_state.summarizer.clear_history()
                st.success("✅ Conversation history cleared")
                st.rerun()
            
            if ask_button:
                if not question.strip():
                    st.warning("⚠️ Please enter a question")
                else:
                    with st.spinner("🤔 Thinking..."):
                        try:
                            answer = safe_ask_question(question)
                            if not answer:
                                st.stop()
                            
                            st.markdown('<div class="success-box">', unsafe_allow_html=True)
                            st.success("✅ Answer Generated")
                            st.markdown('</div>', unsafe_allow_html=True)
                            
                            st.subheader("❓ Question")
                            st.write(question)
                            
                            st.subheader("💬 Answer")
                            st.write(answer)
                            
                        except Exception as e:
                            st.error(f"❌ Error: {str(e)}")
    
    # ========================================================================
    # TAB 4: HISTORY
    # ========================================================================
    with tab4:
        st.header("Analysis History")
        
        if not st.session_state.history:
            st.info("📌 No analysis history yet. Start by summarizing a paper!")
        else:
            st.write(f"**Total Analyses:** {len(st.session_state.history)}")
            
            # Display history in reverse order (newest first)
            for idx, item in enumerate(reversed(st.session_state.history)):
                with st.expander(
                    f"📄 {item['title'][:40]}... | {item['timestamp']} | {item['type']}"
                ):
                    st.write(f"**Type:** {item['type']}")
                    st.write(f"**Summary Preview:**")
                    st.write(item['summary'])
            
            # Clear history button
            if st.button("🗑️ Clear All History", use_container_width=True):
                st.session_state.history = []
                st.success("✅ History cleared")
                st.rerun()


# ============================================================================
# SIDEBAR
# ============================================================================

with st.sidebar:
    anonymous_key = get_or_create_anonymous_key()

    # Information
    st.subheader("ℹ️ About")
    st.write("""
    **Research Paper Summarizer**
    
    
    **Features:**
    - 5 summary types
    - PDF text extraction
    - Key metrics extraction
    - Follow-up Q&A
    - Analysis history
    """)
    
    st.markdown("---")
    
    # Usage Tips
    st.subheader("💡 Tips")
    st.write("""
    1. **Start with Abstract** - Quick overview
    2. **Use Bullet Points** - Key findings
    3. **Technical Summary** - For researchers
    4. **Executive Summary** - For non-experts
    5. **Ask Questions** - Get specific answers
    """)
    
    st.markdown("---")
    
    # Footer
   


# ============================================================================
# FOOTER
# ============================================================================

st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666; font-size: 0.9rem;'>
    <p>Research Paper Summarizer v1.0 |</p>
    </div>
    """,
    unsafe_allow_html=True
)
