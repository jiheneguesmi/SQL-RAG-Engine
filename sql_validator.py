#!/usr/bin/env python3
"""
Interactive SQL Validation Tool
Run this script to manually test and compare LLM aggregation results with your own SQL queries
"""

import sqlite3
import pandas as pd
import os
import sys
import json
from datetime import datetime
from typing import Any, Dict, Optional

# Add your retrieval system to path
sys.path.append('.')  # Adjust path as needed
from retrieval import HybridRetriever  # Import your retrieval system

class InteractiveSQLValidator:
    """Interactive tool for comparing LLM results with manual SQL queries"""
    
    def __init__(self, csv_data_folder: str = "data/", db_path: str = "validation.db"):
        self.csv_folder = csv_data_folder
        self.db_path = db_path
        self.conn = None
        self.retriever = None
        self.loaded_tables = {}
        
        print("🔧 Initializing Interactive SQL Validator...")
        self._setup_database()
        self._setup_retriever()
        self._load_csv_data()
        
    def _setup_database(self):
        """Set up SQLite database"""
        try:
            self.conn = sqlite3.connect(self.db_path)
            self.conn.row_factory = sqlite3.Row
            print(f"✅ Connected to SQLite database: {self.db_path}")
        except Exception as e:
            print(f"❌ Database setup failed: {e}")
            
    def _setup_retriever(self):
        """Initialize your retrieval system"""
        try:
            self.retriever = HybridRetriever()
            print("✅ Retrieval system initialized")
        except Exception as e:
            print(f"❌ Retrieval system setup failed: {e}")
            
    def _load_csv_data(self):
        """Load all CSV files from data folder into SQLite"""
        if not os.path.exists(self.csv_folder):
            print(f"❌ Data folder not found: {self.csv_folder}")
            return
            
        csv_files = [f for f in os.listdir(self.csv_folder) if f.endswith('.csv')]
        
        for csv_file in csv_files:
            try:
                csv_path = os.path.join(self.csv_folder, csv_file)
                table_name = os.path.splitext(csv_file)[0]  # Remove .csv extension
                
                df = pd.read_csv(csv_path)
                df.to_sql(table_name, self.conn, if_exists='replace', index=False)
                
                self.loaded_tables[table_name] = {
                    'rows': len(df),
                    'columns': list(df.columns),
                    'file': csv_file
                }
                
                print(f"✅ Loaded {table_name}: {len(df)} rows, {len(df.columns)} columns")
                
            except Exception as e:
                print(f"❌ Failed to load {csv_file}: {e}")
        
        if self.loaded_tables:
            print(f"\n📊 Loaded {len(self.loaded_tables)} tables successfully")
        else:
            print("⚠️  No tables loaded. Please check your CSV files.")
    
    def show_tables(self):
        """Display available tables and their schemas"""
        if not self.loaded_tables:
            print("No tables available.")
            return
            
        print(f"\n📋 Available Tables ({len(self.loaded_tables)}):")
        print("=" * 80)
        
        for table_name, info in self.loaded_tables.items():
            print(f"\n🔹 {table_name} ({info['file']})")
            print(f"   Rows: {info['rows']}")
            print(f"   Columns: {', '.join(info['columns'])}")
    
    def run_sql_query(self, sql_query: str) -> Dict[str, Any]:
        """Execute a SQL query and return results with timing"""
        if not self.conn:
            return {"error": "Database not connected", "result": None, "time": 0}
        
        start_time = datetime.now()
        try:
            cursor = self.conn.execute(sql_query)
            results = cursor.fetchall()
            
            # Convert results to a more readable format
            if not results:
                formatted_result = "No results"
            elif len(results) == 1 and len(results[0]) == 1:
                # Single value
                formatted_result = results[0][0]
            elif len(results) == 1:
                # Single row
                formatted_result = dict(results[0])
            else:
                # Multiple rows
                formatted_result = [dict(row) for row in results]
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            return {
                "result": formatted_result,
                "time": execution_time,
                "error": None,
                "row_count": len(results)
            }
            
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            return {
                "result": None,
                "time": execution_time,
                "error": str(e),
                "row_count": 0
            }
    
    def run_llm_query(self, natural_query: str) -> Dict[str, Any]:
        """Run the same query through your LLM system"""
        if not self.retriever:
            return {"error": "Retrieval system not available", "result": None, "time": 0}
        
        start_time = datetime.now()
        try:
            # Use your aggregation search
            query_analysis = self.retriever.query_analyzer.analyze_query(natural_query)
            results = self.retriever._aggregation_search(natural_query, query_analysis, 1)
            
            if results:
                # Extract the generated code and columns
                result = results[0]
                llm_output = {
                    "generated_code": result.formatted_content if hasattr(result, 'formatted_content') else result.content,
                    "columns_identified": getattr(result, 'columns_needed', []),
                    "method": result.retrieval_method
                }
            else:
                llm_output = "No LLM results generated"
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            return {
                "result": llm_output,
                "time": execution_time,
                "error": None
            }
            
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            return {
                "result": None,
                "time": execution_time,
                "error": str(e)
            }
    
    def compare_results(self, natural_query: str, sql_query: str):
        """Compare LLM and SQL results side by side"""
        print(f"\n{'='*100}")
        print(f"🔍 COMPARING RESULTS")
        print(f"{'='*100}")
        print(f"Natural Query: {natural_query}")
        print(f"SQL Query: {sql_query}")
        print(f"{'='*100}")
        
        # Run SQL query
        print(f"\n🗄️  SQL EXECUTION:")
        print("-" * 50)
        sql_result = self.run_sql_query(sql_query)
        
        if sql_result["error"]:
            print(f"❌ SQL Error: {sql_result['error']}")
        else:
            print(f"✅ SQL Result: {sql_result['result']}")
            print(f"⏱️  Execution Time: {sql_result['time']:.4f}s")
            if sql_result['row_count'] > 1:
                print(f"📊 Row Count: {sql_result['row_count']}")
        
        # Run LLM query
        print(f"\n🤖 LLM EXECUTION:")
        print("-" * 50)
        llm_result = self.run_llm_query(natural_query)
        
        if llm_result["error"]:
            print(f"❌ LLM Error: {llm_result['error']}")
        else:
            print(f"✅ LLM Result: {llm_result['result']}")
            print(f"⏱️  Execution Time: {llm_result['time']:.4f}s")
        
        # Simple comparison
        print(f"\n📊 COMPARISON:")
        print("-" * 50)
        
        if sql_result["error"] and llm_result["error"]:
            print("❌ Both queries failed")
        elif sql_result["error"]:
            print("⚠️  SQL failed, LLM succeeded")
        elif llm_result["error"]:
            print("⚠️  LLM failed, SQL succeeded")
        else:
            # Both succeeded - let user analyze
            print("✅ Both queries executed successfully")
            print("🔍 Manual comparison required:")
            print(f"   SQL returned: {type(sql_result['result']).__name__}")
            print(f"   LLM returned: {type(llm_result['result']).__name__}")
            
            # Simple type check
            if str(sql_result['result']) == str(llm_result['result']):
                print("🎯 Results appear identical!")
            else:
                print("⚠️  Results differ - manual review needed")
        
        print(f"{'='*100}")
    
    def interactive_session(self):
        """Main interactive session"""
        print("\n" + "="*80)
        print("🚀 INTERACTIVE SQL VALIDATOR")
        print("="*80)
        print("Commands:")
        print("  📋 'tables' - Show available tables")
        print("  🔍 'compare' - Enter natural query + SQL query for comparison")
        print("  🗄️  'sql <query>' - Run SQL query only")
        print("  🤖 'llm <query>' - Run LLM query only")
        print("  ❓ 'help' - Show this help")
        print("  🚪 'quit' - Exit")
        print("="*80)
        
        if not self.loaded_tables:
            print("⚠️  No tables loaded. Please check your data folder.")
            return
        
        while True:
            try:
                user_input = input("\n💬 Enter command: ").strip()
                
                if not user_input:
                    continue
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("👋 Goodbye!")
                    break
                
                elif user_input.lower() == 'tables':
                    self.show_tables()
                
                elif user_input.lower() == 'help':
                    self.show_help()
                
                elif user_input.lower() == 'compare':
                    self.run_comparison_flow()
                
                elif user_input.lower().startswith('sql '):
                    sql_query = user_input[4:].strip()
                    if sql_query:
                        result = self.run_sql_query(sql_query)
                        self.display_single_result("SQL", result)
                
                elif user_input.lower().startswith('llm '):
                    natural_query = user_input[4:].strip()
                    if natural_query:
                        result = self.run_llm_query(natural_query)
                        self.display_single_result("LLM", result)
                
                else:
                    print("❓ Unknown command. Type 'help' for available commands.")
                    
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
    
    def run_comparison_flow(self):
        """Interactive flow for comparing queries"""
        print("\n🔍 COMPARISON MODE")
        print("-" * 50)
        
        # Get natural language query
        natural_query = input("Enter natural language query: ").strip()
        if not natural_query:
            print("❌ Empty query provided")
            return
        
        # Get SQL query
        print("\nNow enter the equivalent SQL query:")
        print("💡 Tip: Use table names from the 'tables' command")
        sql_query = input("Enter SQL query: ").strip()
        if not sql_query:
            print("❌ Empty SQL query provided")
            return
        
        # Run comparison
        self.compare_results(natural_query, sql_query)
    
    def display_single_result(self, query_type: str, result: Dict[str, Any]):
        """Display result from a single query"""
        print(f"\n{query_type} RESULT:")
        print("-" * 30)
        
        if result["error"]:
            print(f"❌ Error: {result['error']}")
        else:
            print(f"✅ Result: {result['result']}")
            print(f"⏱️  Time: {result['time']:.4f}s")
            if 'row_count' in result:
                print(f"📊 Rows: {result['row_count']}")
    
    def show_help(self):
        """Show detailed help"""
        print(f"\n{'='*60}")
        print("📖 HELP - Interactive SQL Validator")
        print(f"{'='*60}")
        print("\n🎯 PURPOSE:")
        print("Compare your LLM aggregation results with manual SQL queries")
        
        print("\n📋 AVAILABLE COMMANDS:")
        print("  tables          - List all loaded tables and columns")
        print("  compare         - Enter natural query + SQL for side-by-side comparison")
        print("  sql <query>     - Execute only SQL query")
        print("  llm <query>     - Execute only LLM query")
        print("  help            - Show this help")
        print("  quit/exit       - Exit the program")
        
        print("\n💡 EXAMPLE WORKFLOW:")
        print("  1. Type 'tables' to see available data")
        print("  2. Type 'compare' to start comparison")
        print("  3. Enter: 'Count all employees'")
        print("  4. Enter: 'SELECT COUNT(*) FROM employees'")
        print("  5. Review side-by-side results")
        
        print("\n📊 SQL TIPS:")
        print("  - Table names match your CSV filenames (without .csv)")
        print("  - Column names are exactly as in your CSV files")
        print("  - Use standard SQL aggregation functions (COUNT, SUM, AVG, etc.)")
        print(f"{'='*60}")
    
    def close(self):
        """Clean up resources"""
        if self.conn:
            self.conn.close()

def main():
    """Main entry point"""
    print("🔧 Starting Interactive SQL Validator...")
    
    # You can customize these paths
    CSV_FOLDER = "data/documnts/"  # Folder containing your CSV files
    DB_PATH = "validation.db"  # SQLite database file
    
    # Check if data folder exists
    if not os.path.exists(CSV_FOLDER):
        print(f"❌ Data folder '{CSV_FOLDER}' not found!")
        print("💡 Please create the folder and add your CSV files, or update CSV_FOLDER path")
        return
    
    try:
        validator = InteractiveSQLValidator(CSV_FOLDER, DB_PATH)
        validator.interactive_session()
    except Exception as e:
        print(f"❌ Failed to start validator: {e}")
    finally:
        if 'validator' in locals():
            validator.close()

if __name__ == "__main__":
    main()