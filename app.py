import streamlit as st
import asyncio
import logging
import json
import pandas as pd
from datetime import datetime
from typing import Dict, Any, List, Optional
import plotly.express as px
 
from feedback_manager import FeedbackManager, StreamlitFeedbackWidget

# PROPERLY DISABLE STREAMLIT MAGIC FUNCTIONS
st.set_option('deprecation.showPyplotGlobalUse', False)

# Try to import your existing modules with flexible handling
try:
    from generation import SmartGenerationOrchestrator, CohereLLMClient, GenerationStrategy
    GENERATION_AVAILABLE = True
except ImportError:
    GENERATION_AVAILABLE = False

try:
    try:
        from retrieval import HybridRetriever
        RETRIEVAL_CLASS = HybridRetriever
    except ImportError:
        try:
            from retrieval import ConversationalRAG
            RETRIEVAL_CLASS = ConversationalRAG
        except ImportError:
            try:
                from retrieval import ConversationalRAG
                RETRIEVAL_CLASS = ConversationalRAG
            except ImportError:
                RETRIEVAL_CLASS = None
    
    RETRIEVAL_AVAILABLE = RETRIEVAL_CLASS is not None
except Exception:
    RETRIEVAL_AVAILABLE = False

try:
    from config import Config
    CONFIG_AVAILABLE = True
except ImportError:
    CONFIG_AVAILABLE = False
    class Config:
        COHERE_API_KEY = "your-cohere-api-key"
        TAVILY_API_KEY = "your-tavily-api-key" 
        CHAT_MODEL = "command-r-plus"
        TEMPERATURE = 0.1
        MAX_ANSWER_LENGTH = 1000

try:
    import cohere
    COHERE_AVAILABLE = True
except ImportError:
    COHERE_AVAILABLE = False

try:
    from tavily import TavilyClient
    TAVILY_AVAILABLE = True
except ImportError:
    TAVILY_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="SQL RAG Query Engine",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced Custom CSS for a professional look
st.markdown("""
<style>
    /* Navigation Bar */
    .navbar {
        background-color: #1f77b4;
        padding: 1rem 2rem;
        color: #fff;
        display: flex;
        align-items: center;
        justify-content: space-between;
        border-radius: 0 0 12px 12px;
        margin-bottom: 2rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    }
    .navbar-logo {
        font-size: 2rem;
        font-weight: bold;
        letter-spacing: 2px;
        display: flex;
        align-items: center;
    }
    .navbar-logo span {
        margin-left: 0.5rem;
    }
    .navbar-links {
        font-size: 1rem;
        font-weight: 500;
    }
    /* Card Containers */
    .card {
        background: #fff;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.07);
        padding: 1.5rem;
        margin-bottom: 1.5rem;
    }
    .main-header {
        font-size: 2.2rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        letter-spacing: 1px;
    }
    .metric-container {
        background-color: #f4f8fb;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        margin: 0.5rem 0;
        text-align: center;
    }
    .source-item {
        background-color: #e8f4fd;
        padding: 0.8rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        border-left: 3px solid #1f77b4;
        font-size: 1rem;
    }
    .feedback-section {
        background: #f9fafb;
        border-radius: 10px;
        padding: 1rem 1.5rem;
        margin-top: 1.5rem;
        box-shadow: 0 1px 4px rgba(0,0,0,0.04);
    }
    .footer {
        text-align: center;
        color: #888;
        font-size: 0.95rem;
        margin-top: 2rem;
        padding-bottom: 1rem;
    }
    .stButton > button {
        background-color: #1f77b4;
        color: #fff;
        font-weight: 600;
        border-radius: 8px;
        padding: 0.5rem 1.2rem;
        transition: background 0.2s;
    }
    .stButton > button:hover {
        background-color: #155a8a;
    }
    # Add this to your existing CSS section
</style>
""", unsafe_allow_html=True)

# Add this to your existing CSS section
st.markdown("""
<style>
    .sql-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        color: white;
    }
    .sql-header {
        display: flex;
        align-items: center;
        margin-bottom: 1rem;
    }
    .sql-status-dot {
        width: 12px;
        height: 12px;
        border-radius: 50%;
        margin-right: 10px;
        animation: pulse 2s infinite;
    }
    .sql-executing {
        background: #ffc107;
    }
    .sql-completed {
        background: #28a745;
    }
    .sql-failed {
        background: #dc3545;
    }
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.5; }
        100% { opacity: 1; }
    }
    .results-table {
        margin-top: 1rem;
        border-radius: 8px;
        overflow: hidden;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# Add navigation bar at the top
st.markdown("""
<div class="navbar">
    <div class="navbar-logo">🤖 <span>SQL RAG Query Engine</span></div>
    <div class="navbar-links">Professional Document Intelligence</div>
</div>
""", unsafe_allow_html=True)

# Claude-style UI CSS
st.markdown("""
<style>
    body, .main, .block-container {
        background: #f6f7f9 !important;
        font-family: 'Segoe UI', 'Roboto', 'Arial', sans-serif;
    }
    .chat-container {
        max-width: 700px;
        margin: 0 auto;
        padding: 2rem 0 6rem 0;
    }
    .chat-bubble {
        background: #fff;
        border-radius: 18px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.07);
        padding: 1.1rem 1.5rem;
        margin-bottom: 1.2rem;
        font-size: 1.15rem;
        line-height: 1.6;
        word-break: break-word;
        max-width: 90%;
    }
    .chat-bubble.user {
        background: #e8f4fd;
        border-bottom-right-radius: 4px;
        margin-left: auto;
        text-align: right;
    }
    .chat-bubble.ai {
        background: #fff;
        border-bottom-left-radius: 4px;
        margin-right: auto;
        text-align: left;
    }
    .chat-meta {
        font-size: 0.95rem;
        color: #888;
        margin-bottom: 0.3rem;
        margin-top: -0.7rem;
    }
    .chat-input-bar {
        position: fixed;
        left: 0; right: 0; bottom: 0;
        background: #fff;
        box-shadow: 0 -2px 12px rgba(0,0,0,0.07);
        padding: 1rem 0.5rem;
        z-index: 100;
    }
    .chat-input-inner {
        max-width: 700px;
        margin: 0 auto;
        display: flex;
        gap: 0.5rem;
    }
    .chat-input-box {
        flex: 1;
        border-radius: 12px;
        border: 1px solid #d1d5db;
        padding: 0.7rem 1rem;
        font-size: 1.1rem;
        background: #f6f7f9;
        outline: none;
    }
    .chat-send-btn {
        background: #1f77b4;
        color: #fff;
        border: none;
        border-radius: 10px;
        padding: 0.7rem 1.3rem;
        font-size: 1.1rem;
        font-weight: 600;
        cursor: pointer;
        transition: background 0.2s;
    }
    .chat-send-btn:hover {
        background: #155a8a;
    }
    .sql-execution-status {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    border-radius: 12px;
    padding: 1rem;
    margin: 1rem 0;
    color: white;
    display: flex;
    align-items: center;
}

.status-dot {
    width: 12px;
    height: 12px;
    border-radius: 50%;
    margin-right: 10px;
    animation: pulse 2s infinite;
}

.status-executing { background: #ffc107; }
.status-completed { background: #28a745; }
.status-failed { background: #dc3545; }

@keyframes pulse {
    0% { opacity: 1; }
    50% { opacity: 0.5; }
    100% { opacity: 1; }
}

</style>
""", unsafe_allow_html=True)


class StreamlitRAGApp:
    """Streamlit application for the SQL RAG Query Engine"""
    
    def __init__(self):
        self.orchestrator = None
        self.feedback_manager = FeedbackManager()
        self.feedback_widget = StreamlitFeedbackWidget(self.feedback_manager)
        self.initialize_session_state()
    
    def initialize_session_state(self):
        """Initialize Streamlit session state variables"""
       
        # Only essential session state variables
        if 'initialized' not in st.session_state:
            st.session_state.initialized = False
        if 'query_history' not in st.session_state:
            st.session_state.query_history = []
        if 'current_response' not in st.session_state:
            st.session_state.current_response = None
    
    def check_system_requirements(self):
        """Check if all required components are available"""
        issues = []
        
        if not GENERATION_AVAILABLE:
            issues.append("❌ generation.py module not found")
        if not RETRIEVAL_AVAILABLE:
            issues.append("❌ retrieval module not found")
        if not COHERE_AVAILABLE:
            issues.append("❌ cohere package not installed")
        if not CONFIG_AVAILABLE:
            issues.append("⚠️ config.py not found (using defaults)")
        if not TAVILY_AVAILABLE:
            issues.append("⚠️ tavily package not installed (web search disabled)")
        
        return issues
    
    @st.cache_resource
    def load_system(_self):
        """Load and cache the RAG system components"""
        try:
            if not GENERATION_AVAILABLE or not RETRIEVAL_AVAILABLE or not COHERE_AVAILABLE:
                return None
            
            retriever = RETRIEVAL_CLASS()
            llm_client = CohereLLMClient()
            
            tavily_client = None
            if TAVILY_AVAILABLE and hasattr(Config, 'TAVILY_API_KEY') and Config.TAVILY_API_KEY:
                try:
                    tavily_client = TavilyClient(api_key=Config.TAVILY_API_KEY)
                except Exception:
                    pass
            
            orchestrator = SmartGenerationOrchestrator(
                retriever=retriever,
                llm_client=llm_client,
                tavily_client=tavily_client
            )
            
            return orchestrator
            
        except Exception as e:
            logger.error(f"System loading error: {e}")
            return None
    
    def render_header(self):
        """Render the application header"""
        st.markdown('<h1 class="main-header">🤖 SQL RAG Query Engine</h1>', unsafe_allow_html=True)
        st.markdown("""
        <div style="text-align: center; margin-bottom: 2rem;">
            <p style="font-size: 1.2rem; color: #666;">
                Intelligent document retrieval and generation with adaptive strategies
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    def render_system_status(self):
        """Render system status and requirements"""
        issues = self.check_system_requirements()
        
        if not issues:
            return True
            
        st.markdown("### 🔧 System Status")
        
        for issue in issues:
            if "❌" in issue:
                st.error(issue)
            else:
                st.warning(issue)
        
        if any("❌" in issue for issue in issues):
            st.markdown("""
            ### 📋 Setup Instructions:
            
            1. **Make sure you have these files in your project:**
               - `generation.py` (with SmartGenerationOrchestrator class)
               - `retrieval.py` or `Retrieval.py` (with your retrieval class)
               - `config.py` (with API keys and settings)
            
            2. **Install required packages:**
               ```bash
               pip install cohere tavily-python streamlit plotly pandas
               ```
            
            3. **Check your imports in generation.py match your actual file structure**
            """)
            return False
        
        return True
    
    def render_sidebar(self):
        """Render the sidebar with configuration and strategy info"""
        with st.sidebar:
            st.header("⚙️ System Status")
            
            components = [
                ("Generation Module", GENERATION_AVAILABLE),
                ("Retrieval Module", RETRIEVAL_AVAILABLE),
                ("Cohere API", COHERE_AVAILABLE),
                ("Config Module", CONFIG_AVAILABLE),
                ("Tavily API", TAVILY_AVAILABLE),
            ]
            
            for name, available in components:
                if available:
                    st.success(f"✅ {name}")
                else:
                    st.error(f"❌ {name}")
            
            st.markdown("---")
            
            if self.orchestrator:
                st.header("🎯 Generation Strategies")
                strategies = {
                    "🔍 RAG Primary": "High relevance + non-temporal",
                    "🔗 RAG Augmented": "Medium relevance",
                    "🧠 Knowledge Primary": "Conceptual-only",
                    "🌐 Web Search": "Temporal-only",
                    "🔄 Hybrid": "High relevance + temporal",
                    "⚠️ Fallback": "Low relevance"
                }
                
                for strategy, description in strategies.items():
                    st.markdown(f"**{strategy}**")
                    st.caption(description)
                
                st.markdown("---")
                
                if hasattr(self.orchestrator, 'HIGH_RELEVANCE_THRESHOLD'):
                    st.subheader("📊 Thresholds")
                    st.metric("High Relevance", f"{self.orchestrator.HIGH_RELEVANCE_THRESHOLD:.2f}")
                    st.metric("Medium Relevance", f"{self.orchestrator.MEDIUM_RELEVANCE_THRESHOLD:.2f}")
    
    def render_query_interface(self):
        """Render Claude-style chat input bar at the bottom"""
        if not self.orchestrator:
            st.error("❌ System not available - please fix the issues above first")
            return "", False
        st.markdown('<div class="chat-input-bar"><div class="chat-input-inner">', unsafe_allow_html=True)
        query = st.text_input(
            "",
            value=st.session_state.get('selected_example', ""),
            placeholder="Type your question...",
            key="query_input",
            label_visibility="collapsed"
        )
        submit_button = st.button("Send", key="send_btn", use_container_width=False)
        st.markdown('</div></div>', unsafe_allow_html=True)
        if st.session_state.get('selected_example'):
            st.session_state.selected_example = ""
        return query, submit_button
    
    def process_query_safe(self, query: str) -> Dict[str, Any]:
        """Enhanced query processing with feedback tracking and LangSmith metrics attachment"""
        
        # Check if we're already processing this exact query
        query_hash = hash(query.strip().lower())
        
        if hasattr(self, '_processing_queries'):
            if query_hash in self._processing_queries:
                logger.warning(f"Query already being processed, skipping: {query}")
                return {"answer": "Query already being processed...", "strategy": "Duplicate", "confidence": 0.0, "query": query}
        else:
            self._processing_queries = set()
        
        # Mark query as being processed
        self._processing_queries.add(query_hash)
        
        try:
            # Initialize session run tracking
            if not hasattr(self, 'session_run_ids'):
                self.session_run_ids = []
            
            logger.info(f"Starting to process query: {query}")
            
            # CRITICAL: Only call the orchestrator once
            if hasattr(self.orchestrator, 'process_query_with_evaluation'):
                logger.info("Using process_query_with_evaluation")
                
                # Create new event loop for this specific query
                try:
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        # If loop is already running, create a new one
                        import concurrent.futures
                        with concurrent.futures.ThreadPoolExecutor() as executor:
                            future = executor.submit(
                                lambda: asyncio.run(self.orchestrator.process_query_with_evaluation(query))
                            )
                            result = future.result(timeout=30)  # 30 second timeout
                    else:
                        result = loop.run_until_complete(self.orchestrator.process_query_with_evaluation(query))
                except Exception as e:
                    logger.error(f"Async execution failed: {e}")
                    # Fallback to sync processing
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    result = loop.run_until_complete(self.orchestrator.process_query_with_evaluation(query))
                    loop.close()
                
                # Attach LangSmith metrics if present in result
                metrics = result.get('metrics')
                if metrics and isinstance(metrics, dict):
                    result['evaluation'] = metrics
                elif metrics and hasattr(metrics, '__dict__'):
                    result['evaluation'] = metrics.__dict__
                
            elif hasattr(self.orchestrator, 'process_query'):
                logger.info("Using process_query fallback")
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                result = loop.run_until_complete(self.orchestrator.process_query(query))
                loop.close()
            else:
                logger.error("No processing method found")
                return {"answer": "No processing method available", "strategy": "Error", "confidence": 0.0, "query": query}
            
            # Track run ID for feedback
            run_id = result.get('langsmith_run_id')
            if run_id and run_id not in self.session_run_ids:
                self.session_run_ids.append(run_id)
            
            # Ensure query is in result
            result['query'] = query
            
            logger.info(f"Query processing completed successfully: {result.get('strategy', 'unknown')}")
            return result
            
        except Exception as e:
            logger.error(f"Query processing failed: {e}")
            return {
                "answer": f"Error processing query: {str(e)}",
                "strategy": "Error",
                "confidence": 0.0,
                "sources": [],
                "reasoning": f"Error: {str(e)}",
                "query": query
            }
        
        finally:
            # Remove query from processing set
            self._processing_queries.discard(query_hash)

            
    def render_evaluation_metrics(self, response: Dict[str, Any]):
        """Render all LangSmith evaluation metrics in a prominent section"""
        evaluation = response.get('evaluation', {})
        if not evaluation:
            return
        st.markdown('''
        <div style="margin-top:1.5rem;background:#f4f8fb;border-radius:10px;padding:1.2rem 1.5rem;box-shadow:0 1px 4px rgba(0,0,0,0.04);">
            <div style="font-size:1.15rem;color:#1f77b4;font-weight:600;margin-bottom:0.7rem;">📊 LangSmith Evaluation Metrics</div>
        ''', unsafe_allow_html=True)
        # Display main metrics in a row
        col1, col2, col3, col4, col5 = st.columns(5)
        metrics = [
            ("Overall Score", evaluation.get("overall_score", 0)),
            ("Relevance", evaluation.get("relevance_score", 0)),
            ("Accuracy", evaluation.get("accuracy_score", 0)),
            ("Completeness", evaluation.get("completeness_score", 0)),
            ("Hallucination Risk", evaluation.get("hallucination_score", 0)),
        ]
        for col, (label, value) in zip([col1, col2, col3, col4, col5], metrics):
            with col:
                color = "#28a745" if value >= 0.7 else "#ffc107" if value >= 0.5 else "#dc3545"
                if label == "Hallucination Risk":
                    color = "#dc3545" if value > 0.3 else "#28a745"
                st.markdown(f'''
                <div style="text-align:center;background:#fff;border-radius:8px;padding:0.7rem 0.2rem;box-shadow:0 1px 2px rgba(0,0,0,0.03);">
                    <div style="font-size:0.9rem;color:#555;font-weight:500;">{label}</div>
                    <div style="font-size:1.3rem;font-weight:700;margin-top:0.3rem;color:{color};">{value:.2f}</div>
                </div>
                ''', unsafe_allow_html=True)
        # Show explanation if available
        explanation = evaluation.get("explanation", "")
        if explanation:
            st.markdown(f'''
            <div style="margin-top:1rem;font-style:italic;color:#666;">
                <strong>Evaluation Details:</strong> {explanation}
            </div>
            ''', unsafe_allow_html=True)
        # Show any additional LangSmith metrics
        extra_keys = [k for k in evaluation.keys() if k not in {"overall_score", "relevance_score", "accuracy_score", "completeness_score", "hallucination_score", "explanation"}]
        if extra_keys:
            st.markdown('<div style="margin-top:1rem;">', unsafe_allow_html=True)
            for k in extra_keys:
                st.markdown(f'<span style="color:#1f77b4;font-weight:600;">{k}:</span> {evaluation[k]}', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    def render_response(self, response: Dict[str, Any]):
        """Render the response with LangSmith metrics at the top, and warn if missing"""
        if not response:
            return
        # Main chat container
        st.markdown('<div class="chat-container">', unsafe_allow_html=True)
        # User query bubble
        query = response.get('query', 'Unknown query')
        st.markdown(f'''
        <div class="chat-bubble user">
            <div class="chat-meta">You asked:</div>
            {query}
        </div>
        ''', unsafe_allow_html=True)
        # AI response bubble  
        answer = response.get('answer', 'No answer provided')
        if response.get('strategy') == 'SQL Aggregation' and 'Analysis:' in answer:
            # Extract only the analysis part
            analysis_start = answer.find('Analysis:')
            if analysis_start != -1:
                answer = answer[analysis_start:]
                
        st.markdown(f'''
        <div class="chat-bubble ai">
            <div class="chat-meta">SQL RAG Query Engine:</div>
            {answer}
        </div>
        ''', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # FIXED: SQL execution display
        if response.get('generated_sql') or response.get('sql_query'):
            self.render_sql_execution_display(response)
        
        # Show LangSmith metrics at the top, or warn if missing
        evaluation = response.get('evaluation', None)
        if evaluation:
            self.render_evaluation_metrics(response)
        else:
            st.warning("No LangSmith metrics found in response. Check orchestrator and evaluation logic.")
            st.expander("Debug: Full Response").write(response)
        # Backend info layout
        st.markdown('<div style="max-width:700px;margin:2rem auto 0 auto;">', unsafe_allow_html=True)
        
        # Strategy metrics
        st.markdown('<div style="width:100%;text-align:center;">', unsafe_allow_html=True)
        metrics = [
            ("Strategy", response.get('strategy', 'Unknown')),
            ("Confidence", f"{response.get('confidence', 0.0):.2%}"),
            ("Sources", len(response.get('sources', [])))
        ]
        metric_html = ""
        for label, value in metrics:
            metric_html += f'''<div style="display:inline-block;width:30%;min-width:120px;margin:0 0.5%;vertical-align:top;background:#f4f8fb;border-radius:8px;padding:0.7rem 0.2rem;box-shadow:0 1px 2px rgba(0,0,0,0.03);">
                <div style="font-size:1.05rem;color:#1f77b4;font-weight:600;">{label}</div>
                <div style="font-size:1.22rem;font-weight:700;margin-top:0.3rem;">{value}</div>
            </div>'''
        st.markdown(metric_html, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        # Analysis and sources
        gen_context = response.get('generation_context', {})
        if gen_context:
            self.render_detailed_analysis(response)
        
        if response.get('sources'):
            self.render_sources(response['sources'])
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Feedback widget
        run_id = response.get('langsmith_run_id')
        feedback_result = self.feedback_widget.render_feedback_widget(
            query=query,
            answer=answer,
            run_id=run_id
        )
        # Show Query History at the end
        if st.session_state.query_history:
            with st.expander("Show Query History", expanded=False):
                for i, entry in enumerate(reversed(st.session_state.query_history[-5:]), 1):
                    st.markdown(f"**Query {i}:** {entry.get('query', 'Unknown')}")
                    st.markdown(f"**Strategy:** {entry.get('strategy', 'Unknown')}")
                    st.markdown(f"**Confidence:** {entry.get('confidence', 0.0):.2%}")
                    st.markdown(f"**Answer:** {entry.get('answer', '')}")
                    st.markdown("---")
        
    def render_detailed_analysis(self, response: Dict[str, Any]):
        """Render detailed analysis of the generation decision"""
        st.subheader("📈 Analysis Details")
        
        gen_context = response.get('generation_context', {})
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            relevance_level = gen_context.get('relevance_level', 'unknown')
            st.metric("Relevance Level", relevance_level.title())
        
        with col2:
            max_score = gen_context.get('max_score', 0.0)
            st.metric("Max Score", f"{max_score:.3f}")
        
        with col3:
            temporal = gen_context.get('needs_current_info', False)
            st.metric("Temporal Need", "Yes" if temporal else "No")
        
        with col4:
            conceptual = gen_context.get('is_conceptual', False)
            st.metric("Conceptual", "Yes" if conceptual else "No")
    
    def render_sources(self, sources: List[str]):
        """Render source information"""
        st.subheader("📚 Sources")
        
        for i, source in enumerate(sources, 1):
            st.markdown(f"""
            <div class="source-item">
                <strong>Source {i}:</strong> {source}
            </div>
            """, unsafe_allow_html=True)
    
    def get_strategy_color(self, strategy: str) -> str:
        """Get color for strategy display"""
        colors = {
            'rag primary': '#28a745',
            'rag augmented': '#007bff',
            'knowledge primary': '#6f42c1',
            'web search': '#fd7e14',
            'hybrid': '#20c997',
            'fallback': '#6c757d',
            'error': '#dc3545'
        }
        
        strategy_lower = strategy.lower()
        for key, color in colors.items():
            if key in strategy_lower:
                return color
        return '#6c757d'
    
    def get_confidence_color(self, confidence: float) -> str:
        """Get color based on confidence level"""
        if confidence >= 0.8:
            return '#28a745'
        elif confidence >= 0.6:
            return '#ffc107'
        else:
            return '#dc3545'
    
    def render_sql_execution_flow(self, response: Dict[str, Any]):
        """Render SQL execution with progressive display"""
        sql_query = response.get('generated_sql')
        sql_details = response.get('sql_details', {})
        
        if not sql_query:
            return
        
        # Step 1: Show SQL Query
        st.markdown("### 🔍 Generated SQL Query")
        st.code(sql_query, language="sql")
        
        # Step 2: Show execution status
        if sql_details.get('success'):
            st.success(f"✅ Query executed successfully in {sql_details.get('execution_time', 0):.3f}s")
            st.info(f"📊 Returned {sql_details.get('rows_returned', 0)} rows")
        elif sql_details.get('error'):
            st.error(f"❌ Query failed: {sql_details.get('error', 'Unknown error')}")
        else:
            st.warning("⏳ Executing query...")
        
        # Step 3: Show raw results if available
        raw_results = response.get('raw_results', [])
        if raw_results and len(raw_results) <= 10:
            st.markdown("### 📋 Query Results")
            st.dataframe(pd.DataFrame(raw_results))
    
    
    def render_sql_execution_display(self, response: Dict[str, Any]):
        """FIXED: Properly display SQL execution results"""
        sql_query = response.get('generated_sql') or response.get('sql_query')
        sql_details = response.get('sql_details', {})
        raw_results = response.get('raw_results', [])
        aggregation_output = response.get('aggregation_output', '')
        
        if not sql_query:
            return
        
        # SQL Query Display
        st.markdown('''
        <div style="margin-top:1.5rem;background:#fff;border-radius:10px;padding:1.2rem 1.5rem;box-shadow:0 1px 4px rgba(0,0,0,0.04);">
            <div style="font-size:1.12rem;color:#1f77b4;font-weight:600;margin-bottom:0.7rem;">Generated SQL Query</div>
        </div>
        ''', unsafe_allow_html=True)
        st.code(sql_query, language="sql")
        
        # Execution Status
        if sql_details.get('success', False):
            execution_time = sql_details.get('execution_time', 0)
            rows_returned = sql_details.get('rows_returned', 0)
            st.success(f"Query executed successfully in {execution_time:.3f}s - {rows_returned} rows returned")
            
            # Show results table if available
            if raw_results:
                st.markdown('''
                <div style="margin-top:1.2rem;background:#f4f8fb;border-radius:10px;padding:1.2rem 1.5rem;box-shadow:0 1px 4px rgba(0,0,0,0.04);">
                    <div style="font-size:1.12rem;color:#1f77b4;font-weight:600;margin-bottom:0.7rem;">Query Results</div>
                </div>
                ''', unsafe_allow_html=True)
                
                # Convert to DataFrame and display
                try:
                    df = pd.DataFrame(raw_results)
                    st.dataframe(df, use_container_width=True)
                except Exception as e:
                    st.error(f"Error displaying results: {e}")
                    st.write(raw_results)
            
            # Show business-friendly output
            if aggregation_output:
                st.markdown('''
                <div style="margin-top:1.2rem;background:#e8f4fd;border-radius:10px;padding:1.2rem 1.5rem;box-shadow:0 1px 4px rgba(0,0,0,0.04);">
                    <div style="font-size:1.12rem;color:#1f77b4;font-weight:600;margin-bottom:0.7rem;">Business Summary</div>
                </div>
                ''', unsafe_allow_html=True)
                st.markdown(aggregation_output)
                
        elif sql_details.get('error'):
            st.error(f"Query execution failed: {sql_details.get('error', 'Unknown error')}")
        else:
            st.warning("Query execution status unknown")
          
    def run(self):
        """FIXED: Main application runner with duplicate prevention"""
        
        if not self.render_system_status():
            return
            
        if not st.session_state.initialized:
            with st.spinner("Loading RAG System..."):
                self.orchestrator = self.load_system()
                st.session_state.initialized = True
        else:
            self.orchestrator = self.load_system()
        
        # Get query input
        query, submit_button = self.render_query_interface()
        
        # CRITICAL FIX: Prevent duplicate processing
        if submit_button and query.strip():
            # Check if this query was just processed
            if (hasattr(st.session_state, 'last_processed_query') and 
                st.session_state.last_processed_query == query.strip()):
                st.warning("This query was just processed. Please wait or try a different query.")
                return
            
            # Mark query as being processed
            st.session_state.last_processed_query = query.strip()
            
            with st.spinner("Processing your query..."):
                try:
                    response = self.process_query_safe(query.strip())
                    st.session_state.current_response = response
                    st.session_state.query_history.append(response)
                    
                    # Clear the last processed query after successful processing
                    if 'last_processed_query' in st.session_state:
                        del st.session_state.last_processed_query
                        
                except Exception as e:
                    logger.error(f"Error in query processing: {e}")
                    st.error(f"An error occurred: {e}")
                    # Clear the processing flag on error too
                    if 'last_processed_query' in st.session_state:
                        del st.session_state.last_processed_query
        
        # Render current response
        if st.session_state.current_response:
            self.render_response(st.session_state.current_response)
            
def main():
    """Main function to run the Streamlit app"""
    try:
        app = StreamlitRAGApp()
        app.run()
    except Exception as e:
        st.error(f"Error running app: {e}")
        logger.error(f"Application error: {e}")

if __name__ == "__main__":
    try:
        # Ensure we're running in Streamlit context
        _ = st.session_state
        main()
    except Exception:
        print("Please run this app with: streamlit run app.py")
