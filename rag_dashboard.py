# rag_dashboard.py

import asyncio
import time
import os
import sys
from datetime import datetime, timedelta
from typing import Dict, List, Any
import json

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import Config
from evaluation import RAGEvaluator

class RAGPerformanceDashboard:
    """Real-time dashboard for RAG system performance"""
    
    def __init__(self, refresh_interval: int = 30):
        self.evaluator = RAGEvaluator()
        self.refresh_interval = refresh_interval
        self.last_update = None
        self.historical_data = []
        
    def clear_screen(self):
        """Clear terminal screen"""
        os.system('cls' if os.name == 'nt' else 'clear')
    
    def print_header(self):
        """Print dashboard header"""
        print("🚀 RAG SYSTEM PERFORMANCE DASHBOARD")
        print("=" * 80)
        print(f"📊 Project: {Config.LANGCHAIN_PROJECT}")
        print(f"🕒 Last Update: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"🔄 Refresh Rate: Every {self.refresh_interval} seconds")
        print("=" * 80)
    
    def get_real_metrics(self) -> Dict[str, Any]:
        """Fetch real metrics from LangSmith"""
        
        if not self.evaluator or not self.evaluator.client:
            return {"error": "LangSmith client not available"}
        
        try:
            # Get recent runs (last 100)
            project_summary = self.evaluator.get_project_summary(limit=100)
            
            # Get runs from last 24 hours for trend analysis
            runs = list(self.evaluator.client.list_runs(
                project_name=Config.LANGCHAIN_PROJECT,
                limit=100
            ))
            
            # Filter recent runs
            now = datetime.now()
            last_24h = now - timedelta(hours=24)
            last_1h = now - timedelta(hours=1)
            
            recent_runs = []
            hourly_runs = []
            
            for run in runs:
                if hasattr(run, 'start_time') and run.start_time:
                    run_time = run.start_time.replace(tzinfo=None) if run.start_time.tzinfo else run.start_time
                    if run_time > last_24h:
                        recent_runs.append(run)
                        if run_time > last_1h:
                            hourly_runs.append(run)
            
            # Calculate metrics
            metrics = {
                "total_runs": len(runs),
                "runs_24h": len(recent_runs),
                "runs_1h": len(hourly_runs),
                "avg_scores": project_summary.get("average_scores", {}),
                "strategy_distribution": project_summary.get("strategy_distribution", {}),
                "recent_runs": recent_runs[:10],  # Last 10 runs
                "error_rate": 0,  # Calculate from runs with errors
                "avg_response_time": 0  # Calculate from run metadata
            }
            
            # Calculate additional metrics from recent runs
            if recent_runs:
                error_count = 0
                response_times = []
                quality_scores = []
                
                for run in recent_runs:
                    # Check for errors
                    if hasattr(run, 'error') and run.error:
                        error_count += 1
                    
                    # Extract metrics from run extra data
                    if hasattr(run, 'extra') and run.extra:
                        extra = run.extra
                        if isinstance(extra, dict):
                            # Response time
                            if 'response_time' in extra:
                                response_times.append(extra['response_time'])
                            
                            # Quality metrics
                            run_metrics = extra.get('metrics', {})
                            if isinstance(run_metrics, dict):
                                overall_score = run_metrics.get('overall_score')
                                if isinstance(overall_score, (int, float)):
                                    quality_scores.append(overall_score)
                
                metrics["error_rate"] = (error_count / len(recent_runs)) * 100
                metrics["avg_response_time"] = sum(response_times) / len(response_times) if response_times else 0
                metrics["avg_quality_24h"] = sum(quality_scores) / len(quality_scores) if quality_scores else 0
            
            return metrics
            
        except Exception as e:
            return {"error": f"Failed to fetch metrics: {str(e)}"}
    
    def display_overview(self, metrics: Dict[str, Any]):
        """Display overview metrics"""
        
        print("\n📊 SYSTEM OVERVIEW")
        print("-" * 40)
        
        if "error" in metrics:
            print(f"❌ Error: {metrics['error']}")
            return
        
        print(f"🔢 Total Runs: {metrics.get('total_runs', 0)}")
        print(f"📈 Last 24h: {metrics.get('runs_24h', 0)} runs")
        print(f"⚡ Last Hour: {metrics.get('runs_1h', 0)} runs")
        print(f"⚠️  Error Rate (24h): {metrics.get('error_rate', 0):.1f}%")
        
        if metrics.get('avg_response_time', 0) > 0:
            print(f"⏱️  Avg Response Time: {metrics.get('avg_response_time', 0):.2f}s")
        
        if metrics.get('avg_quality_24h', 0) > 0:
            print(f"🎯 Avg Quality (24h): {metrics.get('avg_quality_24h', 0):.3f}")
    
    def display_quality_metrics(self, metrics: Dict[str, Any]):
        """Display quality metrics"""
        
        print("\n📈 QUALITY METRICS")
        print("-" * 40)
        
        avg_scores = metrics.get('avg_scores', {})
        
        if not avg_scores:
            print("No quality metrics available yet")
            return
        
        for metric, score in avg_scores.items():
            if isinstance(score, (int, float)):
                # Format metric name
                display_name = metric.replace('_', ' ').title()
                
                # Add status indicator
                if metric == 'hallucination_score':
                    if score < 0.3:
                        status = "🟢 Good"
                    elif score < 0.6:
                        status = "🟡 Fair"
                    else:
                        status = "🔴 Poor"
                    print(f"{display_name}: {score:.3f} ({status})")
                else:
                    if score > 0.7:
                        status = "🟢 Good"
                    elif score > 0.4:
                        status = "🟡 Fair"
                    else:
                        status = "🔴 Poor"
                    print(f"{display_name}: {score:.3f} ({status})")
    
    def display_strategy_distribution(self, metrics: Dict[str, Any]):
        """Display strategy usage distribution"""
        
        print("\n🎯 STRATEGY USAGE")
        print("-" * 40)
        
        strategies = metrics.get('strategy_distribution', {})
        
        if not strategies:
            print("No strategy data available yet")
            return
        
        total_uses = sum(strategies.values())
        
        # Sort by usage
        sorted_strategies = sorted(strategies.items(), key=lambda x: x[1], reverse=True)
        
        for strategy, count in sorted_strategies:
            percentage = (count / total_uses) * 100 if total_uses > 0 else 0
            
            # Create simple bar chart
            bar_length = int(percentage / 5)  # Scale to fit terminal
            bar = "█" * bar_length + "░" * (20 - bar_length)
            
            # Truncate long strategy names
            display_strategy = strategy[:30] + "..." if len(strategy) > 30 else strategy
            
            print(f"{display_strategy:<33} {bar} {count:3d} ({percentage:5.1f}%)")
    
    def display_recent_activity(self, metrics: Dict[str, Any]):
        """Display recent activity"""
        
        print("\n📝 RECENT ACTIVITY")
        print("-" * 40)
        
        recent_runs = metrics.get('recent_runs', [])
        
        if not recent_runs:
            print("No recent activity")
            return
        
        for run in recent_runs[:5]:  # Show last 5
            if hasattr(run, 'start_time') and run.start_time:
                run_time = run.start_time.strftime('%H:%M:%S')
                
                # Get strategy from extra data
                strategy = "Unknown"
                quality_score = "N/A"
                
                if hasattr(run, 'extra') and run.extra and isinstance(run.extra, dict):
                    strategy = run.extra.get('generation_context', {}).get('strategy', 'Unknown')
                    strategy = strategy[:25] + "..." if len(strategy) > 25 else strategy
                    
                    run_metrics = run.extra.get('metrics', {})
                    if isinstance(run_metrics, dict):
                        overall_score = run_metrics.get('overall_score')
                        if isinstance(overall_score, (int, float)):
                            quality_score = f"{overall_score:.2f}"
                
                # Status indicator
                status = "✅"
                if hasattr(run, 'error') and run.error:
                    status = "❌"
                
                print(f"{run_time} {status} {strategy:<30} Quality: {quality_score}")
    
    def display_alerts(self, metrics: Dict[str, Any]):
        """Display system alerts"""
        
        print("\n🚨