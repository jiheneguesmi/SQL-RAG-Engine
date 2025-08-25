import logging
import json
import uuid
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from langsmith import Client, evaluate
from langsmith.evaluation import LangChainStringEvaluator
import cohere
from config import Config
from datetime import datetime
import re
import asyncio

logger = logging.getLogger(__name__)

@dataclass
class EvaluationMetrics:
    """Metrics for answer quality evaluation"""
    hallucination_score: float  # 0-1, lower is better
    relevance_score: float      # 0-1, higher is better
    completeness_score: float   # 0-1, higher is better
    accuracy_score: float       # 0-1, higher is better
    overall_score: float        # 0-1, higher is better
    explanation: str
    timestamp: str = None  # Optional timestamp for when the evaluation was done
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()


@dataclass
class UserFeedback:
    """User feedback data"""
    query: str
    answer: str
    feedback_type: str  # "thumbs_up", "thumbs_down"
    feedback_details: Optional[str] = None
    timestamp: str = None
    session_id: Optional[str] = None
    run_id: Optional[str] = None  # linked to LangSmith run if available
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()

class RAGEvaluator:
    """RAG system evaluator with LangSmith integration"""
    
    def __init__(self):
        try:
            self.client = Client(api_key=Config.LANGSMITH_API_KEY)
            self.llm_client = cohere.Client(Config.COHERE_API_KEY)
            self.project_name = Config.LANGSMITH_PROJECT
            logger.info(f"RAGEvaluator initialized with LangSmith project: {self.project_name}")
        except Exception as e:
            logger.error(f"Failed to initialize RAGEvaluator: {e}")
            self.client = None
            self.llm_client = None
        
    def detect_hallucination(self, query: str, answer: str, source_materials: List[str]) -> float:
        """Detect hallucinations by comparing answer against source materials"""
        
        if not self.llm_client:
            logger.warning("LLM client not available, returning default hallucination score")
            return 0.3
        
        source_text = "\n".join(source_materials[:3])  # Limit source text length
        
        hallucination_prompt = f"""
        You are an expert fact-checker. Compare the generated answer against the source materials to detect hallucinations.
        
        Query: {query}
        
        Generated Answer: {answer[:1000]}  # Limit answer length
        
        Source Materials:
        {source_text[:2000]}  # Limit source length
        
        Task: Identify any claims in the generated answer that are NOT supported by the source materials.
        
        Rate the hallucination level from 0.0 (no hallucinations) to 1.0 (severe hallucinations).
        
        Return ONLY a JSON object:
        {{
            "hallucination_score": 0.0,
            "unsupported_claims": ["claim1"],
            "explanation": "Brief explanation"
        }}
        """
        
        try:
            response = self.llm_client.chat(
                message=hallucination_prompt,
                model=Config.CHAT_MODEL,
                temperature=0.1,
                max_tokens=500
            )
            
            response_text = response.text.strip()
            logger.info(f"Raw LLM response: {response_text[:200]}...")  # Debug log
            
            # Try multiple JSON extraction methods
            result = None
            
            # Method 1: Direct JSON parse
            try:
                result = json.loads(response_text)
            except json.JSONDecodeError:
                pass
            
        except Exception as e:
            logger.error(f"Hallucination detection failed: {e}")
            return 0.3  # Default moderate score
    
    def score_answer_quality(self, query: str, answer: str, source_materials: List[str]) -> EvaluationMetrics:
        """Comprehensive answer quality scoring"""
        
        if not self.llm_client:
            logger.warning("LLM client not available, returning default metrics")
            return EvaluationMetrics(0.5, 0.5, 0.5, 0.5, 0.5, "LLM client unavailable")
        
        # Limit input sizes to avoid token limits
        source_text = "\n".join(source_materials[:3])[:2000] if source_materials else "No source materials provided"
        limited_answer = answer[:1000]
        
        quality_prompt = f"""
    You are an expert evaluator for RAG (Retrieval Augmented Generation) systems. 
    Evaluate this response carefully and provide realistic scores.

    Query: {query}

    Generated Answer: {limited_answer}

    Source Materials:
    {source_text}

    Please evaluate the answer on these dimensions (scale 0.0-1.0):

    1. RELEVANCE (0.0-1.0): How well does the answer address the specific query?
    - 1.0 = Perfectly addresses all aspects of the query
    - 0.5 = Partially relevant, misses some aspects
    - 0.0 = Completely irrelevant

    2. COMPLETENESS (0.0-1.0): Does the answer fully address the query?
    - 1.0 = Complete, comprehensive answer
    - 0.5 = Partial answer, missing some details
    - 0.0 = Incomplete or superficial

    3. ACCURACY (0.0-1.0): Are the facts correct based on source materials?
    - 1.0 = All facts are accurate and verifiable
    - 0.5 = Mostly accurate with minor issues
    - 0.0 = Contains factual errors

    4. HALLUCINATION (0.0-1.0): Are there unsupported claims?
    - 0.0 = No hallucinations, all claims supported
    - 0.5 = Some minor unsupported details
    - 1.0 = Major hallucinations or false information

    BE CRITICAL AND REALISTIC. Most answers should have some room for improvement.
    Perfect scores (1.0) should be rare and only for exceptional responses.

    Return ONLY valid JSON format:
    {{
    "relevance_score": 0.8,
    "completeness_score": 0.7,
    "accuracy_score": 0.9,
    "hallucination_score": 0.1,
    "explanation": "Brief explanation here"
}}
    """
        
        try:
            response = self.llm_client.chat(
                message=quality_prompt,
                model=Config.CHAT_MODEL,
                temperature=0.2,  # Slightly higher for more varied responses
                max_tokens=800    # More tokens for detailed evaluation
            )
            
            # Clean and parse JSON response
            response_text = response.text.strip()
            
            # More robust JSON extraction
            # Look for JSON block between curly braces
            import re
            json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', response_text)
            if json_match:
                response_text = json_match.group(0)
            
            # Remove any markdown formatting
            response_text = re.sub(r'```json\s*|\s*```', '', response_text)
            
            result = json.loads(response_text)
            
            # Validate scores are in correct range
            def clamp_score(score, default=0.5):
                try:
                    score = float(score)
                    return max(0.0, min(1.0, score))
                except:
                    return default
            
            # Calculate overall score as weighted average
            relevance = clamp_score(result.get("relevance_score", 0.5))
            completeness = clamp_score(result.get("completeness_score", 0.5))
            accuracy = clamp_score(result.get("accuracy_score", 0.5))
            hallucination = clamp_score(result.get("hallucination_score", 0.3))
            
            # Weighted overall score (hallucination penalty)
            overall = (relevance * 0.3 + completeness * 0.3 + accuracy * 0.3 + (1.0 - hallucination) * 0.1)
            
            return EvaluationMetrics(
                hallucination_score=hallucination,
                relevance_score=relevance,
                completeness_score=completeness,
                accuracy_score=accuracy,
                overall_score=overall,
                explanation=result.get("explanation", "Evaluation completed")
            )
            
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing failed: {e}")
            logger.error(f"Raw response: {response_text}")
            return EvaluationMetrics(0.3, 0.5, 0.5, 0.5, 0.5, f"JSON parsing failed: {str(e)}")
            
        except Exception as e:
            logger.error(f"Quality scoring failed: {e}")
            return EvaluationMetrics(0.3, 0.5, 0.5, 0.5, 0.5, f"Evaluation failed: {str(e)}")
        
    def log_interaction(self, query: str, answer: str, sources: List[str], 
                       metrics: EvaluationMetrics, generation_context: Dict[str, Any]) -> Optional[str]:
        """Log interaction to LangSmith for tracking"""
        
        if not self.client:
            logger.warning("LangSmith client not available, skipping logging")
            return None
        
        try:
            # Generate unique run ID
            run_id = str(uuid.uuid4())
            
            # Prepare inputs and outputs
            inputs = {
                "query": query,
                "strategy": generation_context.get("strategy", "unknown"),
                "confidence": generation_context.get("confidence", 0.0),
                "relevance_level": generation_context.get("relevance_level", "unknown")
            }
            
            outputs = {
                "answer": answer[:1000],  # Limit answer length
                "sources_count": len(sources),
                "retrieval_methods": generation_context.get("retrieval_method_counts", {})
            }
            
            # Create run with extra metadata
            extra_data = {
                "sources": sources[:5],  # Limit sources
                "metrics": {
                    "hallucination_score": metrics.hallucination_score,
                    "relevance_score": metrics.relevance_score,
                    "completeness_score": metrics.completeness_score,
                    "accuracy_score": metrics.accuracy_score,
                    "overall_score": metrics.overall_score,
                    "explanation": metrics.explanation
                },
                "generation_context": {
                    "strategy": generation_context.get("strategy"),
                    "confidence": generation_context.get("confidence"),
                    "relevance_level": generation_context.get("relevance_level"),
                    "needs_current_info": generation_context.get("needs_current_info"),
                    "is_conceptual": generation_context.get("is_conceptual"),
                    "max_score": generation_context.get("max_score"),
                    "content_relevance_score": generation_context.get("content_relevance_score")
                },
                "timestamp": datetime.now().isoformat(),
                "system_version": "1.0"
            }
            
            # Create run
            run = self.client.create_run(
                name="rag_generation",
                run_type="llm",  # Specify run type
                inputs=inputs,
                outputs=outputs,
                extra=extra_data,
                project_name=self.project_name,
                id=run_id
            )
            
            # Add scores as feedback
            self._add_automatic_feedback(run_id, metrics)
            
            logger.info(f"Successfully logged interaction to LangSmith: {run_id}")
            return run_id
            
        except Exception as e:
            logger.error(f"LangSmith logging failed: {e}")
            return None
    
    def _add_automatic_feedback(self, run_id: str, metrics: EvaluationMetrics):
        """Add automatic feedback scores to the run"""
        
        try:
            feedback_data = [
                ("hallucination", 1.0 - metrics.hallucination_score),  # Invert so higher is better
                ("relevance", metrics.relevance_score),
                ("completeness", metrics.completeness_score),
                ("accuracy", metrics.accuracy_score),
                ("overall_quality", metrics.overall_score)
            ]
            
            for key, score in feedback_data:
                self.client.create_feedback(
                    run_id=run_id,
                    key=key,
                    score=score,
                    comment=metrics.explanation if key == "overall_quality" else None
                )
                
        except Exception as e:
            logger.error(f"Failed to add automatic feedback: {e}")
    
    def log_user_feedback(self, feedback: UserFeedback):
        """Log user feedback to LangSmith"""
        
        if not self.client:
            logger.warning("LangSmith client not available, skipping user feedback")
            return
        
        try:
            # Calculate score from feedback type
            score = 1.0 if feedback.feedback_type == "thumbs_up" else 0.0
            
            # Create feedback
            self.client.create_feedback(
                run_id=feedback.run_id,
                key="user_satisfaction",
                score=score,
                comment=feedback.feedback_details,
                source_info={
                    "query": feedback.query[:500],
                    "timestamp": feedback.timestamp,
                    "session_id": feedback.session_id,
                    "feedback_type": feedback.feedback_type
                }
            )
            
            logger.info(f"User feedback logged for run: {feedback.run_id}")
            
        except Exception as e:
            logger.error(f"User feedback logging failed: {e}")
    
    def create_dataset(self, name: str, queries_and_answers: List[Dict[str, Any]]):
        """Create a dataset for evaluation"""
        
        if not self.client:
            logger.warning("LangSmith client not available")
            return None
        
        try:
            examples = []
            for item in queries_and_answers:
                examples.append({
                    "inputs": {"query": item["query"]},
                    "outputs": {"expected_answer": item.get("expected_answer", "")}
                })
            
            dataset = self.client.create_dataset(
                dataset_name=name,
                description=f"RAG evaluation dataset created on {datetime.now().isoformat()}",
                examples=examples
            )
            
            logger.info(f"Created dataset: {name}")
            return dataset
            
        except Exception as e:
            logger.error(f"Dataset creation failed: {e}")
            return None
    
    def get_run_metrics(self, run_id: str) -> Dict[str, Any]:
        """Get metrics for a specific run"""
        
        if not self.client:
            return {}
        
        try:
            run = self.client.read_run(run_id)
            feedbacks = list(self.client.list_feedback(run_ids=[run_id]))
            
            metrics = {}
            for feedback in feedbacks:
                metrics[feedback.key] = feedback.score
            
            return {
                "run_id": run_id,
                "inputs": run.inputs,
                "outputs": run.outputs,
                "metrics": metrics,
                "extra": run.extra
            }
            
        except Exception as e:
            logger.error(f"Failed to get run metrics: {e}")
            return {}
    
    def get_project_summary(self, limit: int = 100) -> Dict[str, Any]:
        """Get summary statistics for the project"""
        
        if not self.client:
            return {}
        
        try:
            runs = list(self.client.list_runs(
                project_name=self.project_name,
                limit=limit
            ))
            
            if not runs:
                return {"message": "No runs found"}
            
            # Collect metrics
            total_runs = len(runs)
            strategies = {}
            avg_scores = {}
            
            for run in runs:
                # Count strategies
                strategy = run.extra.get("generation_context", {}).get("strategy", "unknown")
                strategies[strategy] = strategies.get(strategy, 0) + 1
                
                # Collect scores
                metrics = run.extra.get("metrics", {})
                for metric, value in metrics.items():
                    if metric not in avg_scores:
                        avg_scores[metric] = []
                    if isinstance(value, (int, float)):
                        avg_scores[metric].append(value)
            
            # Calculate averages
            for metric in avg_scores:
                if avg_scores[metric]:
                    avg_scores[metric] = sum(avg_scores[metric]) / len(avg_scores[metric])
                else:
                    avg_scores[metric] = 0
            
            return {
                "total_runs": total_runs,
                "strategy_distribution": strategies,
                "average_scores": avg_scores,
                "project_url": f"https://smith.langchain.com/projects/{self.project_name}"
            }
            
        except Exception as e:
            logger.error(f"Failed to get project summary: {e}")
            return {"error": str(e)}