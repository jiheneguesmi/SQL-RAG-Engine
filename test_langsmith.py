# interactive_rag_monitor.py

import asyncio
import sys
import os
from typing import Dict, List, Any, Optional
from datetime import datetime
import json

# Add your project root to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import Config
from evaluation import RAGEvaluator, UserFeedback, EvaluationMetrics
from generation import CohereLLMClient, SmartGenerationOrchestrator  
from retrieval import HybridRetriever

class InteractiveRAGMonitor:
    """Interactive RAG monitoring system with real evaluation"""
    
    def __init__(self):
        self.retriever = HybridRetriever()
        self.llm_client = CohereLLMClient()
        self.orchestrator = SmartGenerationOrchestrator(self.retriever, self.llm_client)
        self.evaluator = RAGEvaluator()
        self.session_id = f"interactive_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.orchestrator.session_id = self.session_id
        
        # Track session statistics
        self.session_stats = {
            "queries_processed": 0,
            "positive_feedback": 0,
            "negative_feedback": 0,
            "strategies_used": {},
            "avg_scores": {},
            "start_time": datetime.now()
        }
        
    def print_welcome(self):
        """Print welcome message"""
        print("🎯 Interactive RAG Monitoring System")
        print("=" * 60)
        print(f"📊 Session ID: {self.session_id}")
        print(f"🔗 LangSmith Project: {Config.LANGSMITH_PROJECT}")
        print("=" * 60)
        print()
        print("This system will:")
        print("✅ Process your queries through the full RAG pipeline")
        print("📈 Evaluate answers using LLM-based metrics")
        print("💬 Collect your feedback for continuous improvement")
        print("📊 Track performance metrics in real-time")
        print("🌐 Log everything to LangSmith for analysis")
        print()
        print("Commands:")
        print("  • Enter any query to test the system")
        print("  • 'stats' - Show session statistics")
        print("  • 'help' - Show available commands")
        print("  • 'quit' - Exit and show final report")
        print("=" * 60)
    
    def print_help(self):
        """Print help information"""
        print("\n📚 Available Commands:")
        print("  • <query> - Ask any question to test the RAG system")
        print("  • 'stats' - Show current session statistics")
        print("  • 'strategies' - Show strategy distribution")
        print("  • 'langsmith' - Show LangSmith project info")
        print("  • 'examples' - Show example queries you can try")
        print("  • 'help' - Show this help message")
        print("  • 'quit' or 'exit' - Exit with session report")
        print()
        
    def show_examples(self):
        """Show example queries"""
        examples = [
            "show cartons with taxes foncieres",
            "count total documents by user",
            "what is document management?",
            "show users with email addresses",
            "latest news about document archiving",
            "how many cartons contain property documents?",
            "explain the RAG retrieval process",
            "find all documents from 2023"
        ]
        
        print("\n💡 Example Queries to Try:")
        for i, example in enumerate(examples, 1):
            print(f"  {i}. {example}")
        print()
    
    async def process_query_with_feedback(self, query: str) -> Dict[str, Any]:
        """Process query and collect comprehensive feedback"""
        
        print(f"\n🔍 Processing: '{query}'")
        print("⏳ Analyzing query and generating response...")
        
        # Process query with full evaluation
        start_time = datetime.now()
        result = await self.orchestrator.process_query_with_evaluation(query)
        processing_time = (datetime.now() - start_time).total_seconds()
        
        # Update session stats
        self.session_stats["queries_processed"] += 1
        strategy = result.get('strategy', 'unknown')
        self.session_stats["strategies_used"][strategy] = \
            self.session_stats["strategies_used"].get(strategy, 0) + 1
        
        # Display results
        self._display_query_results(result, processing_time)
        
        # Collect user feedback
        feedback_data = self._collect_user_feedback(query, result)
        
        # Update session metrics
        self._update_session_metrics(result, feedback_data)
        
        return {
            'result': result,
            'feedback': feedback_data,
            'processing_time': processing_time
        }
    
    def _display_query_results(self, result: Dict[str, Any], processing_time: float):
        """Display query results in a nice format"""
        
        print("\n" + "="*60)
        print("📋 QUERY RESULTS")
        print("="*60)
        
        # Strategy and performance info
        print(f"🎯 Strategy: {result.get('strategy', 'Unknown')}")
        print(f"⚡ Processing Time: {processing_time:.2f}s")
        print(f"🎲 Confidence: {result.get('confidence', 0):.2f}")
        
        # Sources info
        sources = result.get('sources', [])
        if sources:
            print(f"📚 Sources: {len(sources)} documents retrieved")
            
        # SQL info if available
        if 'sql_details' in result:
            sql_info = result['sql_details']
            print(f"🗄️  SQL Query Executed: {sql_info.get('rows_returned', 0)} rows returned")
        
        print("\n📝 ANSWER:")
        print("-" * 40)
        answer = result.get('answer', 'No answer generated')
        print(answer)
        print("-" * 40)
        
        # Evaluation metrics
        evaluation = result.get('evaluation', {})
        if evaluation:
            print("\n📊 EVALUATION METRICS:")
            print(f"  🎯 Overall Score: {evaluation.get('overall_score', 0):.2f}")
            print(f"  📈 Relevance: {evaluation.get('relevance_score', 0):.2f}")
            print(f"  ✅ Accuracy: {evaluation.get('accuracy_score', 0):.2f}")
            print(f"  📋 Completeness: {evaluation.get('completeness_score', 0):.2f}")
            print(f"  ⚠️  Hallucination Risk: {evaluation.get('hallucination_score', 0):.2f} (lower is better)")
        
        # LangSmith run info
        run_id = result.get('langsmith_run_id')
        if run_id:
            print(f"\n🌐 LangSmith Run: {run_id[:12]}...")
            print(f"   View: https://smith.langchain.com/projects/{Config.LANGSMITH_PROJECT}")
        
        print("="*60)
    
    def _collect_user_feedback(self, query: str, result: Dict[str, Any]) -> Optional[UserFeedback]:
        """Collect detailed user feedback"""
        
        print("\n💬 FEEDBACK COLLECTION")
        print("-" * 30)
        
        # Get overall satisfaction
        while True:
            feedback_type = input("How would you rate this answer? (👍 good / 👎 bad / ⏭️ skip): ").strip().lower()
            
            if feedback_type in ['👍', 'good', 'g', 'thumbs_up', '1']:
                feedback_type = 'thumbs_up'
                break
            elif feedback_type in ['👎', 'bad', 'b', 'thumbs_down', '0']:
                feedback_type = 'thumbs_down'
                break
            elif feedback_type in ['⏭️', 'skip', 's', '']:
                return None
            else:
                print("Please enter: 👍/good/g, 👎/bad/b, or ⏭️/skip/s")
        
        # Get detailed feedback
        feedback_details = input("Optional: What specifically was good/bad? ").strip()
        
        # Create feedback object
        feedback = UserFeedback(
            query=query,
            answer=result.get('answer', ''),
            feedback_type=feedback_type,
            feedback_details=feedback_details if feedback_details else None,
            session_id=self.session_id,
            run_id=result.get('langsmith_run_id')
        )
        
        # Log to LangSmith
        if self.evaluator:
            self.evaluator.log_user_feedback(feedback)
            print(f"✅ Feedback logged to LangSmith")
        
        return feedback
    
    def _update_session_metrics(self, result: Dict[str, Any], feedback: Optional[UserFeedback]):
        """Update running session metrics"""
        
        # Update feedback counts
        if feedback:
            if feedback.feedback_type == 'thumbs_up':
                self.session_stats["positive_feedback"] += 1
            else:
                self.session_stats["negative_feedback"] += 1
        
        # Update evaluation metrics
        evaluation = result.get('evaluation', {})
        if evaluation:
            for metric in ['overall_score', 'relevance_score', 'accuracy_score', 'completeness_score', 'hallucination_score']:
                if metric not in self.session_stats["avg_scores"]:
                    self.session_stats["avg_scores"][metric] = []
                
                score = evaluation.get(metric, 0)
                self.session_stats["avg_scores"][metric].append(score)
    
    def show_session_stats(self):
        """Show current session statistics"""
        
        print("\n📊 SESSION STATISTICS")
        print("=" * 50)
        
        stats = self.session_stats
        session_duration = (datetime.now() - stats["start_time"]).total_seconds() / 60
        
        print(f"⏱️  Session Duration: {session_duration:.1f} minutes")
        print(f"🔢 Queries Processed: {stats['queries_processed']}")
        
        # Feedback stats
        total_feedback = stats["positive_feedback"] + stats["negative_feedback"]
        if total_feedback > 0:
            satisfaction_rate = (stats["positive_feedback"] / total_feedback) * 100
            print(f"👍 Positive Feedback: {stats['positive_feedback']} ({satisfaction_rate:.1f}%)")
            print(f"👎 Negative Feedback: {stats['negative_feedback']}")
        else:
            print("💬 No feedback collected yet")
        
        # Strategy distribution
        if stats["strategies_used"]:
            print(f"\n🎯 Strategies Used:")
            for strategy, count in stats["strategies_used"].items():
                percentage = (count / stats["queries_processed"]) * 100
                print(f"   {strategy}: {count} times ({percentage:.1f}%)")
        
        # Average scores
        if stats["avg_scores"]:
            print(f"\n📈 Average Evaluation Scores:")
            for metric, scores in stats["avg_scores"].items():
                if scores:
                    avg_score = sum(scores) / len(scores)
                    print(f"   {metric.replace('_', ' ').title()}: {avg_score:.3f}")
        
        print("=" * 50)
    
    def show_strategy_details(self):
        """Show detailed strategy information"""
        
        print("\n🎯 STRATEGY ANALYSIS")
        print("=" * 50)
        
        if not self.session_stats["strategies_used"]:
            print("No strategies used yet. Try running some queries!")
            return
        
        total_queries = self.session_stats["queries_processed"]
        
        for strategy, count in self.session_stats["strategies_used"].items():
            percentage = (count / total_queries) * 100
            
            print(f"\n📋 {strategy}:")
            print(f"   Usage: {count}/{total_queries} queries ({percentage:.1f}%)")
            
            # Strategy explanation
            explanations = {
                "RAG Primary (High Relevance + Non-Temporal)": "Uses retrieved documents with high confidence",
                "RAG Augmented (Medium Relevance)": "Combines retrieval with general knowledge",
                "Knowledge Primary (Conceptual Only)": "Relies on pre-trained knowledge",
                "Web Search (Temporal Only)": "Uses web search for current information",
                "Hybrid (High Relevance + Temporal)": "Combines retrieval and web search",
                "SQL Aggregation": "Generates and executes SQL queries",
                "Fallback (Otherwise)": "Default strategy when others don't apply"
            }
            
            explanation = explanations.get(strategy, "Custom strategy")
            print(f"   Description: {explanation}")
        
        print("=" * 50)
    
    def show_langsmith_info(self):
        """Show LangSmith project information"""
        
        print(f"\n🌐 LANGSMITH PROJECT INFO")
        print("=" * 50)
        print(f"Project: {Config.LANGSMITH_PROJECT}")
        print(f"Session ID: {self.session_id}")
        print(f"Dashboard: https://smith.langchain.com/projects/{Config.LANGSMITH_PROJECT}")
        
        if self.evaluator and self.evaluator.client:
            try:
                # Get recent project summary
                project_summary = self.evaluator.get_project_summary(limit=50)
                
                if 'total_runs' in project_summary:
                    print(f"\n📊 Project Summary (Last 50 runs):")
                    print(f"   Total Runs: {project_summary['total_runs']}")
                    
                    if 'average_scores' in project_summary:
                        print(f"   Average Scores:")
                        for metric, score in project_summary['average_scores'].items():
                            if isinstance(score, (int, float)):
                                print(f"     {metric.replace('_', ' ').title()}: {score:.3f}")
                    
                    if 'strategy_distribution' in project_summary:
                        print(f"   Strategy Distribution:")
                        for strategy, count in project_summary['strategy_distribution'].items():
                            print(f"     {strategy}: {count} runs")
                
            except Exception as e:
                print(f"   ⚠️  Could not fetch project summary: {e}")
        
        print("=" * 50)
    
    def generate_final_report(self):
        """Generate final session report"""
        
        print("\n" + "="*70)
        print("📋 FINAL SESSION REPORT")
        print("="*70)
        
        stats = self.session_stats
        session_duration = (datetime.now() - stats["start_time"]).total_seconds() / 60
        
        print(f"📅 Session: {self.session_id}")
        print(f"⏱️  Duration: {session_duration:.1f} minutes")
        print(f"🔢 Total Queries: {stats['queries_processed']}")
        
        # Performance summary
        if stats["queries_processed"] > 0:
            queries_per_minute = stats["queries_processed"] / session_duration
            print(f"⚡ Avg Speed: {queries_per_minute:.1f} queries/minute")
        
        # User satisfaction
        total_feedback = stats["positive_feedback"] + stats["negative_feedback"]
        if total_feedback > 0:
            satisfaction_rate = (stats["positive_feedback"] / total_feedback) * 100
            print(f"😊 User Satisfaction: {satisfaction_rate:.1f}% ({stats['positive_feedback']}/{total_feedback})")
        
        # Quality metrics
        if stats["avg_scores"]:
            print(f"\n📊 Average Quality Metrics:")
            for metric, scores in stats["avg_scores"].items():
                if scores:
                    avg_score = sum(scores) / len(scores)
                    
                    # Add interpretation
                    if metric == 'hallucination_score':
                        status = "🟢 Low" if avg_score < 0.3 else "🟡 Medium" if avg_score < 0.6 else "🔴 High"
                        print(f"   {metric.replace('_', ' ').title()}: {avg_score:.3f} ({status} Risk)")
                    else:
                        status = "🟢 Good" if avg_score > 0.7 else "🟡 Fair" if avg_score > 0.4 else "🔴 Poor"
                        print(f"   {metric.replace('_', ' ').title()}: {avg_score:.3f} ({status})")
        
        # Top strategies
        if stats["strategies_used"]:
            print(f"\n🎯 Most Used Strategies:")
            sorted_strategies = sorted(stats["strategies_used"].items(), key=lambda x: x[1], reverse=True)
            for strategy, count in sorted_strategies[:3]:
                percentage = (count / stats["queries_processed"]) * 100
                print(f"   {strategy}: {count} times ({percentage:.1f}%)")
        
        # Recommendations
        print(f"\n💡 Recommendations:")
        if total_feedback == 0:
            print("   • Collect more user feedback to improve accuracy")
        elif satisfaction_rate < 70:
            print("   • Review low-rated responses to identify improvement areas")
        
        if stats["avg_scores"].get('hallucination_score', []):
            avg_hallucination = sum(stats["avg_scores"]['hallucination_score']) / len(stats["avg_scores"]['hallucination_score'])
            if avg_hallucination > 0.4:
                print("   • Review retrieval quality to reduce hallucinations")
        
        print(f"\n🌐 View detailed analytics at:")
        print(f"   https://smith.langchain.com/projects/{Config.LANGSMITH_PROJECT}")
        
        print("="*70)
    
    async def run_interactive_session(self):
        """Main interactive session loop"""
        
        # Initialize and show welcome
        self.print_welcome()
        
        try:
            while True:
                user_input = input("\n🤖 Enter query or command: ").strip()
                
                if not user_input:
                    continue
                
                # Handle commands
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("\n👋 Ending session...")
                    break
                
                elif user_input.lower() == 'help':
                    self.print_help()
                
                elif user_input.lower() == 'stats':
                    self.show_session_stats()
                
                elif user_input.lower() == 'strategies':
                    self.show_strategy_details()
                
                elif user_input.lower() == 'langsmith':
                    self.show_langsmith_info()
                
                elif user_input.lower() == 'examples':
                    self.show_examples()
                
                else:
                    # Process as query
                    await self.process_query_with_feedback(user_input)
        
        except KeyboardInterrupt:
            print("\n\n⏹️  Session interrupted by user")
        
        except Exception as e:
            print(f"\n❌ Session error: {e}")
            
        finally:
            # Always show final report
            self.generate_final_report()


async def main():
    """Main entry point"""
    
    print("🚀 Starting Interactive RAG Monitoring System...")
    
    # Verify configuration
    if not Config.LANGSMITH_API_KEY:
        print("❌ LANGSMITH API key not configured. Please set LANGSMITH_API_KEY in your config.")
        return
    
    if not Config.COHERE_API_KEY:
        print("❌ Cohere API key not configured. Please set COHERE_API_KEY in your config.")
        return
    
    # Create and run monitor
    monitor = InteractiveRAGMonitor()
    await monitor.run_interactive_session()


if __name__ == "__main__":
    asyncio.run(main())