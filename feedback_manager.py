# feedback_manager.py
import logging
import uuid
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
from datetime import datetime
import streamlit as st
from langsmith import Client
from config import Config

logger = logging.getLogger(__name__)

@dataclass
class FeedbackEntry:
    """Simple feedback entry"""
    feedback_id: str
    query: str
    answer: str
    rating: str  # "positive", "negative", "neutral"
    comment: Optional[str] = None
    run_id: Optional[str] = None
    timestamp: str = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()

class FeedbackManager:
    """Simplified feedback manager with LangSmith integration"""
    
    def __init__(self):
        self.client = None
        try:
            self.client = Client(api_key=Config.LANGSMITH_API_KEY)
            logger.info("FeedbackManager initialized with LangSmith")
        except Exception as e:
            logger.error(f"Failed to initialize LangSmith client: {e}")
    
    def submit_feedback(self, query: str, answer: str, rating: str, 
                       comment: Optional[str] = None, run_id: Optional[str] = None) -> bool:
        """Submit feedback to LangSmith"""
        try:
            feedback_id = str(uuid.uuid4())
            
            feedback_entry = FeedbackEntry(
                feedback_id=feedback_id,
                query=query,
                answer=answer,
                rating=rating,
                comment=comment,
                run_id=run_id
            )
            
            if self.client and run_id:
                # Submit to LangSmith
                score = self._rating_to_score(rating)
                
                self.client.create_feedback(
                    run_id=run_id,
                    key="user_satisfaction",
                    score=score,
                    comment=comment,
                    source_info={
                        "feedback_id": feedback_id,
                        "timestamp": feedback_entry.timestamp,
                        "query": query[:500],  # Truncate long queries
                        "rating": rating
                    }
                )
                
                logger.info(f"Feedback submitted to LangSmith: {feedback_id}")
                return True
            else:
                logger.warning(f"LangSmith client not available or no run_id. Feedback stored locally: {feedback_id}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to submit feedback: {e}")
            return False
    
    def _rating_to_score(self, rating: str) -> float:
        """Convert rating to numeric score"""
        rating_map = {
            "positive": 1.0,
            "negative": 0.0,
            "neutral": 0.5
        }
        return rating_map.get(rating.lower(), 0.5)
    
    def get_feedback_stats(self, run_ids: List[str]) -> Dict[str, Any]:
        """Get feedback statistics for given runs"""
        if not self.client:
            return {}
        
        try:
            feedbacks = list(self.client.list_feedback(run_ids=run_ids))
            
            if not feedbacks:
                return {"total_feedback": 0}
            
            positive = sum(1 for f in feedbacks if f.score >= 0.7)
            negative = sum(1 for f in feedbacks if f.score <= 0.3)
            neutral = len(feedbacks) - positive - negative
            
            return {
                "total_feedback": len(feedbacks),
                "positive": positive,
                "negative": negative,  
                "neutral": neutral,
                "satisfaction_rate": (positive / len(feedbacks)) * 100 if feedbacks else 0
            }
        except Exception as e:
            logger.error(f"Failed to get feedback stats: {e}")
            return {}


# Updated Streamlit feedback component
class StreamlitFeedbackWidget:
    """Clean Streamlit feedback widget"""
    
    def __init__(self, feedback_manager: FeedbackManager):
        self.feedback_manager = feedback_manager
    
    def render_feedback_widget(self, query: str, answer: str, run_id: Optional[str] = None) -> Optional[str]:
        """Render a clean feedback widget"""
        
        # Create unique key for this response
        widget_key = f"feedback_{hash(query + answer[:50])}"
        
        st.markdown("---")
        st.subheader("📝 How was this response?")
        
        # Simple button-based feedback
        col1, col2, col3 = st.columns(3)
        
        feedback_given = None
        
        with col1:
            if st.button("👍 Good", key=f"{widget_key}_good", use_container_width=True):
                feedback_given = "positive"
        
        with col2:
            if st.button("👎 Not Good", key=f"{widget_key}_bad", use_container_width=True):
                feedback_given = "negative"
        
        with col3:
            if st.button("😐 Neutral", key=f"{widget_key}_neutral", use_container_width=True):
                feedback_given = "neutral"
        
        # If feedback was given, collect optional comment
        if feedback_given:
            st.session_state[f"{widget_key}_feedback_type"] = feedback_given
            
            # Show thank you and optional comment
            emoji_map = {"positive": "👍", "negative": "👎", "neutral": "😐"}
            st.success(f"Thank you for your feedback! {emoji_map[feedback_given]}")
            
            comment = st.text_input(
                "Optional: Tell us more",
                key=f"{widget_key}_comment",
                placeholder="What specifically was good/bad?",
                max_chars=500
            )
            
            if st.button("Submit Feedback", key=f"{widget_key}_submit", type="primary"):
                success = self.feedback_manager.submit_feedback(
                    query=query,
                    answer=answer,
                    rating=feedback_given,
                    comment=comment if comment.strip() else None,
                    run_id=run_id
                )
                
                if success:
                    st.success("✅ Feedback submitted successfully!")
                    # Clear the feedback state
                    if f"{widget_key}_feedback_type" in st.session_state:
                        del st.session_state[f"{widget_key}_feedback_type"]
                    return "submitted"
                else:
                    st.warning("⚠️ Feedback saved locally (LangSmith not available)")
                    return "saved_locally"
        
        return None


# Updated app.py integration
def integrate_feedback_into_app():
    """Example of how to integrate this into your main app"""
    
    # In your StreamlitRAGApp.__init__
    def __init__(self):
        # ... existing code ...
        self.feedback_manager = FeedbackManager()
        self.feedback_widget = StreamlitFeedbackWidget(self.feedback_manager)
    
    # Replace your render_response method's feedback section with:
    def render_response(self, response: Dict[str, Any]):
        """Updated render_response with clean feedback"""
        
        # ... existing response rendering code ...
        
        # Clean feedback section
        query = response.get("query", "")
        answer = response.get("answer", "")
        run_id = response.get("langsmith_run_id")
        
        # Render feedback widget
        feedback_result = self.feedback_widget.render_feedback_widget(
            query=query,
            answer=answer,
            run_id=run_id
        )
        
        # Optional: Show feedback stats
        if feedback_result == "submitted":
            st.rerun()  # Refresh to clear feedback state


# Enhanced evaluation.py integration
class EnhancedRAGEvaluator:
    """Enhanced evaluator with better feedback integration"""
    
    def __init__(self):
        self.client = None
        self.feedback_manager = FeedbackManager()
        try:
            self.client = Client(api_key=Config.LANGSMITH_API_KEY)
        except Exception as e:
            logger.error(f"Failed to initialize LangSmith client: {e}")
    
    def log_interaction_with_feedback_ready(self, query: str, answer: str, sources: List[str], 
                                          metrics: 'EvaluationMetrics', 
                                          generation_context: Dict[str, Any]) -> Optional[str]:
        """Log interaction and prepare for feedback collection"""
        
        if not self.client:
            return None
        
        try:
            run_id = str(uuid.uuid4())
            
            # Enhanced logging with feedback readiness
            inputs = {
                "query": query,
                "strategy": generation_context.get("strategy", "unknown"),
                "confidence": generation_context.get("confidence", 0.0)
            }
            
            outputs = {
                "answer": answer[:1000],
                "sources_count": len(sources)
            }
            
            extra_data = {
                "sources": sources[:5],
                "metrics": asdict(metrics),
                "generation_context": generation_context,
                "feedback_ready": True,  # Mark as ready for feedback
                "timestamp": datetime.now().isoformat()
            }
            
            # Create run
            run = self.client.create_run(
                name="rag_generation",
                run_type="llm",
                inputs=inputs,
                outputs=outputs,
                extra=extra_data,
                project_name=Config.LANGSMITH_PROJECT,
                id=run_id
            )
            
            logger.info(f"Interaction logged to LangSmith: {run_id}")
            return run_id
            
        except Exception as e:
            logger.error(f"LangSmith logging failed: {e}")
            return None
    
    def get_session_feedback_summary(self, session_id: str) -> Dict[str, Any]:
        """Get feedback summary for a session"""
        try:
            # Query runs for this session
            runs = list(self.client.list_runs(
                project_name=Config.LANGSMITH_PROJECT,
                limit=100,
                filter=f'extra.session_id = "{session_id}"'
            ))
            
            if not runs:
                return {"message": "No runs found for session"}
            
            run_ids = [run.id for run in runs]
            return self.feedback_manager.get_feedback_stats(run_ids)
            
        except Exception as e:
            logger.error(f"Failed to get session feedback: {e}")
            return {"error": str(e)}


# Simple usage example for testing
def test_feedback_system():
    """Test the feedback system"""
    
    feedback_manager = FeedbackManager()
    
    # Simulate feedback submission
    success = feedback_manager.submit_feedback(
        query="What is machine learning?",
        answer="Machine learning is a subset of AI...",
        rating="positive",
        comment="Very helpful explanation!",
        run_id="test-run-123"
    )
    
    print(f"Feedback submission: {'Success' if success else 'Failed'}")
    
    # Test feedback stats
    stats = feedback_manager.get_feedback_stats(["test-run-123"])
    print(f"Feedback stats: {stats}")


if __name__ == "__main__":
    test_feedback_system()