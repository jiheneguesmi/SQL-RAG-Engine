import os
import re
import json
import logging
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime
import pandas as pd
import numpy as np
from langchain.docstore.document import Document
from langchain_cohere import CohereEmbeddings
from langchain_community.vectorstores import FAISS
from config import Config
import cohere
import ast
from pathlib import Path
import sqlite3
import tempfile


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    """Structured result from retrieval system"""

    content: str
    metadata: Dict[str, Any]
    score: float
    source_table: str
    retrieval_method: str
    relevance_explanation: str = ""
    formatted_content: str = ""  # Add formatted content for better display

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary with parsed columns"""
        if " | " not in self.content:
            return {
                "content": self.content,
                "source": self.source_table,
                "score": self.score,
            }

        row_dict = {}
        parts = self.content.split(" | ")
        for part in parts:
            if ":" in part:
                key, value = part.split(":", 1)
                row_dict[key.strip()] = value.strip()

        row_dict["_source"] = self.source_table
        row_dict["_score"] = self.score
        row_dict["_method"] = self.retrieval_method

        return row_dict

@dataclass
class SQLGenerationResult:
    """Result from SQL generation for aggregation queries"""
    sql_query: str
    explanation: str
    tables_used: List[str]
    columns_used: List[str]
    aggregation_type: str
    confidence: float
    estimated_result: str = ""
    errors: List[str] = field(default_factory=list)
    
@dataclass
class QueryAnalysis:
    """Analysis of user query for optimal retrieval strategy"""

    original_query: str
    query_type: str
    entities: List[str]
    keywords: List[str]
    potential_columns: List[str]
    confidence: float


class QueryAnalyzer:
    """Analyzes queries to determine optimal retrieval strategy"""

    def __init__(self):
        # Enhanced column patterns for business/organizational data
        self.column_patterns = {
            "identity": [
                "id",
                "name",
                "title",
                "nom",
                "prenom",
                "firstname",
                "lastname",
                "codebarre",
                "barcode",
            ],
            "contact": ["email", "phone", "telephone", "address", "adresse"],
            "temporal": [
                "date",
                "time",
                "year",
                "month",
                "created",
                "updated",
                "timestamp",
                "2011",
                "2012",
            ],
            "financial": [
                "price",
                "cost",
                "amount",
                "salary",
                "budget",
                "revenue",
                "tax",
                "taxes",
            ],
            "location": [
                "city",
                "country",
                "region",
                "location",
                "ville",
                "pays",
                "emplacement",
            ],
            "status": ["status", "state", "active", "enabled", "type", "category"],
            "description": [
                "description",
                "comment",
                "notes",
                "details",
                "summary",
                "carton",
                "document",
            ],
            "document_types": [
                "pdf",
                "doc",
                "excel",
                "foncier",
                "property",
                "carton",
                "dossier",
            ],
        }

    def analyze_query(self, query: str) -> QueryAnalysis:
        """Analyze query to determine retrieval strategy"""
        query_lower = query.lower()

        # Extract potential entities
        entities = self._extract_entities(query)

        # Extract keywords
        keywords = self._extract_keywords(query)

        # Determine query type
        query_type = self._classify_query_type(query_lower)

        # Map to potential columns
        potential_columns = self._map_to_columns(query_lower, keywords)

        # Calculate confidence
        confidence = self._calculate_confidence(entities, keywords, potential_columns)

        return QueryAnalysis(
            original_query=query,
            query_type=query_type,
            entities=entities,
            keywords=keywords,
            potential_columns=potential_columns,
            confidence=confidence,
        )

    def _extract_entities(self, query: str) -> List[str]:
        """Extract potential entities from query"""
        entities = []

        # Email addresses
        email_pattern = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"
        entities.extend(re.findall(email_pattern, query))

        # Barcodes and IDs (alphanumeric codes)
        barcode_pattern = r"\b[A-Z0-9]{6,}\b"
        entities.extend(re.findall(barcode_pattern, query))

        user_id_patterns = [
        r'\b[A-Z]{2,6}\d{4,8}\b',    # APYR5460, DATA1234, etc.
        r'\b[A-Z]{3,}[0-9]{3,}\b',   # More flexible alphanumeric
        r'\b[A-Z0-9]{6,12}\b'        # General alphanumeric codes
    ]
        for pattern in user_id_patterns:
            matches = re.findall(pattern, query)
            entities.extend(matches)

        # Document/Barcode IDs 
        barcode_pattern = r"\b[A-Z]?\d{9,15}\b"  # P056186473 or 056186473
        entities.extend(re.findall(barcode_pattern, query))
     
        
        # Phone numbers
        phone_pattern = (
            r"\b(?:\+?1[-.\s]?)?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}\b"
        )
        entities.extend(re.findall(phone_pattern, query))

        # Years (4-digit numbers)
        year_pattern = r"\b(19|20)\d{2}\b"
        entities.extend(re.findall(year_pattern, query))

        # Other numbers (potentially IDs, amounts)
        number_pattern = r"\b\d{3,}\b"  # At least 3 digits
        entities.extend(re.findall(number_pattern, query))

        # Names (capitalized words, but filter common words)
        capitalized = re.findall(r"\b[A-Z][a-z]+\b", query)
        common_words = {
            "Show",
            "Find",
            "List",
            "Get",
            "From",
            "For",
            "All",
            "Document",
            "File",
        }
        entities.extend([word for word in capitalized if word not in common_words])

        # Quoted strings
        quoted = re.findall(r'"([^"]*)"', query)
        entities.extend(quoted)

        return list(set(entities))

    def _extract_keywords(self, query: str) -> List[str]:
        """Extract meaningful keywords from query"""
        stop_words = {
            "the",
            "a",
            "an",
            "and",
            "or",
            "but",
            "in",
            "on",
            "at",
            "to",
            "for",
            "of",
            "with",
            "by",
            "is",
            "are",
            "was",
            "were",
            "be",
            "been",
            "being",
            "have",
            "has",
            "had",
            "do",
            "does",
            "did",
            "will",
            "would",
            "could",
            "should",
            "may",
            "might",
            "must",
            "can",
            "this",
            "that",
            "these",
            "those",
            "i",
            "you",
            "he",
            "she",
            "it",
            "we",
            "they",
            "me",
            "him",
            "her",
            "us",
            "them",
            "my",
            "your",
            "his",
            "her",
            "its",
            "our",
            "their",
            "what",
            "where",
            "when",
            "why",
            "how",
            "who",
            "which",
            "all",
            "any",
            "some",
            "more",
            "most",
        }

        # Split and clean words
        words = re.findall(r"\b[a-zA-Z0-9]{2,}\b", query.lower())
        keywords = [word for word in words if word not in stop_words and len(word) > 1]

        return keywords

    def _classify_query_type(self, query: str) -> str:
        """Classify the type of query"""
        if any(
            indicator in query
            for indicator in ["find", "search", "look for", "get", "who is", "what is"]
        ):
            if any(comp in query for comp in ["compare", "vs", "versus", "difference"]):
                return "comparison"
            return "specific"

        if any(
            agg in query
            for agg in [
                "count",
                "sum",
                "total",
                "average",
                "how many",
                "list all",
                "show all",
            ]
        ):
            return "aggregation"

        if any(
            comp in query
            for comp in ["compare", "vs", "versus", "better", "worse", "different"]
        ):
            return "comparison"

        return "general"

    def _map_to_columns(self, query: str, keywords: List[str]) -> List[str]:
        """Map query terms to likely column names"""
        potential_columns = []

        for category, columns in self.column_patterns.items():
            for column in columns:
                if column in query or any(
                    keyword == column or column in keyword or keyword in column
                    for keyword in keywords
                ):
                    potential_columns.append(column)

        return potential_columns

    def _calculate_confidence(
        self, entities: List[str], keywords: List[str], potential_columns: List[str]
    ) -> float:
        """Calculate confidence in query analysis"""
        base_score = 0.2

        # More entities = higher confidence
        entity_score = min(0.4, len(entities) * 0.1)

        # More keywords = higher confidence
        keyword_score = min(0.2, len(keywords) * 0.05)

        # Column matches = higher confidence
        column_score = min(0.2, len(potential_columns) * 0.08)

        return min(1.0, base_score + entity_score + keyword_score + column_score)

@dataclass
class QueryRequest:
    """Structured query request with configuration options"""
    
    query: str
    top_k: int = 5
    retrieval_strategy: str = "auto"  # auto, semantic, structured, hybrid
    display_mode: str = "smart"  # smart, full, compact
    table_filter: Optional[str] = None  # Filter by specific table
    min_score: float = 0.1  # Minimum relevance score
    include_metadata: bool = True
    explain_results: bool = False


class HybridRetriever:
    """Advanced retrieval system combining semantic and structured search"""

    def __init__(
    self, vector_store_path: str = None, embeddings_model: CohereEmbeddings = None
):
        self.vector_store_path = vector_store_path or Config.VECTOR_DB_PATH
        self.embeddings = embeddings_model or CohereEmbeddings(
            model=Config.EMBEDDING_MODEL, cohere_api_key=Config.COHERE_API_KEY
        )
        self.vector_store = None
        self.query_analyzer = QueryAnalyzer()
        # Add Cohere LLM client for aggregation queries
        self.llm_client = cohere.Client(Config.COHERE_API_KEY)
        self.llm_model = getattr(Config, "CHAT_MODEL", "command")
        self.llm_temperature = getattr(Config, "TEMPERATURE", 0.3)
        self.llm_max_tokens = getattr(Config, "MAX_ANSWER_LENGTH", 128)

        self._load_vector_store()
        self.table_schemas = {}
        self._analyze_table_schemas(use_metadata=False)

    def _load_vector_store(self):
        """Load the FAISS vector store"""
        try:
            if os.path.exists(self.vector_store_path):
                self.vector_store = FAISS.load_local(
                    self.vector_store_path,
                    self.embeddings,
                    allow_dangerous_deserialization=True,
                )
                logger.info(f"Vector store loaded from {self.vector_store_path}")
            else:
                logger.error(f"Vector store not found at {self.vector_store_path}")
        except Exception as e:
            logger.error(f"Failed to load vector store: {e}")
            
    def _load_metadata_definitions(self):
        """Load table and column metadata definitions"""
        try:
            metadata_file = Path("table_column_metadata.json")
            if metadata_file.exists():
                with open(metadata_file, 'r', encoding='utf-8') as f:
                    self.metadata_definitions = json.load(f)
                logger.info("Loaded metadata definitions")
            else:
                logger.warning("No metadata definitions file found")
                self.metadata_definitions = {"table_names": {}, "column_names": {}}
        except Exception as e:
            logger.error(f"Failed to load metadata definitions: {e}")
            self.metadata_definitions = {"table_names": {}, "column_names": {}}


    def _analyze_table_schemas(self, use_metadata=False):
        """Analyze schemas of all tables in the vector store"""
        if not self.vector_store:
            return
        # Ensure metadata definitions are loaded only for aggregation queries
        
        if use_metadata:
            self._load_metadata_definitions()
        try:
            # Get more documents for better schema analysis
            all_docs = self.vector_store.similarity_search("", k=2000)

            tables = {}
            for doc in all_docs:
                source = doc.metadata.get("source", "unknown")
                if source not in tables:
                    if use_metadata:
                        table_metadata = self._get_column_metadata(source)
                        tables[source] = {
                            "columns": set(),
                            "sample_content": [],
                            "column_types": {},
                            "sample_values": {},
                            "display_name": table_metadata.get("display_name", source),
                            "description": table_metadata.get("description", ""),
                            "languages": table_metadata.get("languages", {}),
                            "column_metadata": {}
                            }
                    else:
                        tables[source] = {
                            "columns": set(),
                            "sample_content": [],
                            "column_types": {},
                            "sample_values": {},
                
                        }

                # Enhanced column extraction
                content = doc.page_content
                if " | " in content:
                    parts = content.split(" | ")
                    for part in parts:
                        if ":" in part:
                            column_name = part.split(":")[0].strip()
                            column_value = part.split(":", 1)[1].strip()

                            tables[source]["columns"].add(column_name)
                            
                            if use_metadata and column_name not in tables[source]["column_metadata"]:
                                col_metadata = self._get_column_metadata(column_name)
                                tables[source]["column_metadata"][column_name] = col_metadata

                            # Store sample values for each column
                            if column_name not in tables[source]["sample_values"]:
                                tables[source]["sample_values"][column_name] = set()
                            tables[source]["sample_values"][column_name].add(
                                column_value[:100]
                            )

                # Store more sample content
                if len(tables[source]["sample_content"]) < 10:
                    tables[source]["sample_content"].append(content[:500])

            # Convert sets to lists
            for table in tables:
                tables[table]["columns"] = list(tables[table]["columns"])
                for col in tables[table]["sample_values"]:
                    tables[table]["sample_values"][col] = list(
                        tables[table]["sample_values"][col]
                    )[:5]

            self.table_schemas = tables
            logger.info(f"Analyzed schemas for {len(tables)} tables")

        except Exception as e:
            logger.error(f"Failed to analyze table schemas: {e}")
        
    

    def retrieve_all_columns(self, query: str, top_k: int = 5) -> List[RetrievalResult]:
        """Retrieve results showing ALL available columns"""
        return self.retrieve(
            query, top_k, retrieval_strategy="auto", display_mode="full"
        )

    def _format_result_content(
        self, content: str, query_analysis: QueryAnalysis, display_mode: str = "smart"
    ) -> str:
        """Format result content to show most relevant information"""
        if " | " not in content:
            return content

        parts = content.split(" | ")
        row_dict = {}

        # Parse content into dictionary
        for part in parts:
            if ":" in part:
                key, value = part.split(":", 1)
                row_dict[key.strip()] = value.strip()

        # Show ALL columns when display_mode is "full"
        if display_mode == "full":
            formatted_parts = []
            for key, value in row_dict.items():
                formatted_parts.append(f"{key}: {value}")
            return " | ".join(formatted_parts)

        # Original smart formatting logic for other modes
        always_show = [
            "id",
            "name",
            "nom",
            "prenom",
            "firstname",
            "lastname",
            "codebarre",
            "barcode",
            "title",
        ]
        query_relevant = []
        other_important = []

        query_lower = query_analysis.original_query.lower()

        for key, value in row_dict.items():
            key_lower = key.lower()
            value_lower = value.lower()

            # Always show identity columns first
            if key_lower in [col.lower() for col in always_show]:
                continue

            # Skip metadata columns unless specifically requested
            if (
                key_lower in ["filename", "insert_date", "update_date"]
                and key_lower not in query_lower
            ):
                continue

            # Check if column is query-relevant
            is_relevant = False
            if key_lower in query_lower:
                is_relevant = True
            else:
                # Check if value contains any query keywords or entities
                for entity in query_analysis.entities:
                    if entity.lower() in value_lower:
                        is_relevant = True
                        break
                if not is_relevant:
                    for keyword in query_analysis.keywords:
                        if keyword in key_lower or keyword in value_lower:
                            is_relevant = True
                            break

            if is_relevant:
                query_relevant.append(key)
            else:
                other_important.append(key)

        # Select columns to display (max 12 for smart mode)
        display_columns = []

        # Add identity columns first
        for key in row_dict.keys():
            if key.lower() in [col.lower() for col in always_show]:
                display_columns.append(key)

        # Add query-relevant columns
        display_columns.extend(query_relevant)

        # Fill remaining slots with other important columns
        remaining = 12 - len(display_columns)
        display_columns.extend(other_important[:remaining])

        # Format selected columns
        formatted_parts = []
        for col in display_columns:
            if col in row_dict:
                value = row_dict[col]
                formatted_parts.append(f"{col}: {value}")

        return " | ".join(formatted_parts)

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        retrieval_strategy: str = "auto",
        display_mode: str = "smart",
    ) -> List[RetrievalResult]:
        """Main retrieval function with multiple strategies"""

        if not self.vector_store:
            logger.error("Vector store not available")
            return []

        query_analysis = self.query_analyzer.analyze_query(query)
        logger.info(
            f"Query analysis: {query_analysis.query_type}, confidence: {query_analysis.confidence:.2f}"
        )

        # Aggregation strategy detection
        if query_analysis.query_type == "aggregation":
            return self._aggregation_search(query, query_analysis, top_k)

        if retrieval_strategy == "auto":
            retrieval_strategy = self._choose_strategy(query_analysis)

        results = []
        try:
            if retrieval_strategy == "semantic":
                results = self._semantic_search(query, query_analysis, top_k)
            elif retrieval_strategy == "structured":
                results = self._structured_search(query, query_analysis, top_k)
            elif retrieval_strategy == "hybrid":
                results = self._hybrid_search(query, query_analysis, top_k)
            else:
                results = self._hybrid_search(query, query_analysis, top_k)
        except Exception as e:
            logger.error(f"Retrieval failed: {e}")
            try:
                results = self._semantic_search(query, query_analysis, top_k)
            except Exception as e2:
                logger.error(f"Fallback retrieval also failed: {e2}")
                return []
        # Post-process results to remove duplicates and improve formatting
        return self._post_process_results(results, query_analysis, top_k, display_mode)


    def _structured_search(
        self, query: str, query_analysis: QueryAnalysis, top_k: int
    ) -> List[RetrievalResult]:
        """Structured search with robust similarity normalization and scoring"""
        try:
            initial_k = min(top_k * 8, 100)
            docs = self.vector_store.similarity_search_with_score(query, k=initial_k)
            scored_results = []
            for doc, base_distance in docs:
                content_lower = doc.page_content.lower()
                metadata = doc.metadata
                # Normalize base_distance to similarity in [0,1]
                # If base_distance is cosine distance, similarity = 1 - base_distance
                # If base_distance is L2, use similarity = 1.0 / (1.0 + base_distance)
                # Here, we use 1 - base_distance, but clamp to [0,1]
                semantic_similarity = max(0.0, min(1.0, 1 - base_distance))
                structured_score = 0.0
                explanation_parts = []
                entity_matches = 0
                total_entity_score = 0
                for entity in query_analysis.entities:
                    if entity.lower() in content_lower:
                        entity_matches += 1
                        if any(field in content_lower for field in ["codebarre:", "id:", "name:", "nom:"]):
                            total_entity_score += 2.0
                        else:
                            total_entity_score += 1.0
                        explanation_parts.append(f"Entity '{entity}'")
                if query_analysis.entities:
                    entity_score = total_entity_score / len(query_analysis.entities)
                    structured_score += entity_score * 0.5
                keyword_matches = 0
                for keyword in query_analysis.keywords:
                    if keyword in content_lower:
                        keyword_matches += 1
                        explanation_parts.append(f"Keyword '{keyword}'")
                if query_analysis.keywords:
                    keyword_score = keyword_matches / len(query_analysis.keywords)
                    structured_score += keyword_score * 0.3
                column_matches = 0
                for column in query_analysis.potential_columns:
                    if f"{column.lower()}:" in content_lower:
                        column_matches += 1
                        explanation_parts.append(f"Column '{column}'")
                if query_analysis.potential_columns:
                    column_score = column_matches / len(query_analysis.potential_columns)
                    structured_score += column_score * 0.2
                # Combine scores
                if structured_score > 0:
                    final_score = (structured_score * 0.1) + (semantic_similarity * 0.9)
                else:
                    final_score = semantic_similarity * 0.8
                final_score = max(0.0, min(1.0, final_score))
                if final_score > 0.05:
                    result = RetrievalResult(
                        content=doc.page_content,
                        metadata=metadata,
                        score=final_score,
                        source_table=metadata.get("source", "unknown"),
                        retrieval_method="structured",
                        relevance_explanation=(
                            "; ".join(explanation_parts)
                            if explanation_parts
                            else "Semantic match"
                        ),
                    )
                    scored_results.append(result)
            scored_results.sort(key=lambda x: x.score, reverse=True)
            return scored_results
        except Exception as e:
            logger.error(f"Structured search failed: {e}")
            return []

    def _hybrid_search(
        self, query: str, query_analysis: QueryAnalysis, top_k: int
    ) -> List[RetrievalResult]:
        """Enhanced hybrid search"""
        try:
            # Get results from both methods
            semantic_results = self._semantic_search(query, query_analysis, top_k * 3)
            structured_results = self._structured_search(
                query, query_analysis, top_k * 3
            )

            # Merge results with better deduplication
            combined_results = {}

            # Add semantic results
            for result in semantic_results:
                key = self._generate_result_key(result)
                combined_results[key] = result
                combined_results[key].retrieval_method = "hybrid_semantic"

            # Add/update with structured results
            for result in structured_results:
                key = self._generate_result_key(result)
                if key in combined_results:
                    # Weighted average favoring higher scores
                    existing_score = combined_results[key].score
                    new_score = result.score
                    combined_results[key].score = (
                        max(existing_score, new_score) * 0.7
                        + min(existing_score, new_score) * 0.3
                    )
                    combined_results[key].retrieval_method = "hybrid_both"
                    combined_results[key].relevance_explanation = (
                        f"{combined_results[key].relevance_explanation}; {result.relevance_explanation}"
                    )
                else:
                    combined_results[key] = result
                    combined_results[key].retrieval_method = "hybrid_structured"

            # Sort and return
            final_results = list(combined_results.values())
            final_results.sort(key=lambda x: x.score, reverse=True)

            return final_results

        except Exception as e:
            logger.error(f"Hybrid search failed: {e}")
            return self._semantic_search(query, query_analysis, top_k)

    def _generate_result_key(self, result: RetrievalResult) -> str:
        """Generate a unique key for result deduplication"""
        # Use table name and a hash of the core content
        content = result.content
        if " | " in content:
            # Extract meaningful part (skip metadata)
            content_parts = content.split(" | ")
            data_parts = [
                part
                for part in content_parts
                if not part.startswith(("filename:", "insert_date:", "update_date:"))
            ]
            content = " | ".join(data_parts[:5])  # Use first 5 meaningful parts

        return f"{result.source_table}_{hash(content[:300])}"
    def _post_process_results(
        self,
        results: List[RetrievalResult],
        query_analysis: QueryAnalysis,
        top_k: int,
        display_mode: str = "smart",
    ) -> List[RetrievalResult]:
        """Post-process results to remove duplicates and improve display"""

        # Remove duplicates based on content similarity
        unique_results = []
        seen_content = set()

        for result in results:
            # Create a hash of the core content (without metadata prefixes)
            core_content = result.content
            if " | " in core_content:
                # Extract just the data part, skip filename and metadata
                content_parts = core_content.split(" | ")
                data_parts = [
                    part
                    for part in content_parts
                    if not part.startswith(
                        ("filename:", "insert_date:", "update_date:")
                    )
                ]
                core_content = " | ".join(data_parts)

            content_hash = hash(
                core_content[:200]
            )  # Use first 200 chars for similarity

            if content_hash not in seen_content:
                seen_content.add(content_hash)

                # Format the content for better display
                result.formatted_content = self._format_result_content(
                    result.content, query_analysis, display_mode
                )
                unique_results.append(result)

        # Re-sort by score and limit to top_k
        unique_results.sort(key=lambda x: x.score, reverse=True)
        return unique_results[:top_k]

    def _choose_strategy(self, query_analysis: QueryAnalysis) -> str:
        """Choose optimal retrieval strategy"""
        # If we have specific entities (names, IDs, barcodes), use structured
        if query_analysis.confidence > 0.6 and len(query_analysis.entities) > 0:
            return "structured"

        # If medium confidence, use hybrid
        if query_analysis.confidence > 0.4:
            return "hybrid"

        return "semantic"

    def _semantic_search(
        self, query: str, query_analysis: QueryAnalysis, top_k: int
    ) -> List[RetrievalResult]:
        """Enhanced semantic similarity search"""
        try:
            # Get more results initially for better filtering
            initial_k = min(top_k * 5, 50)
            docs = self.vector_store.similarity_search_with_score(query, k=initial_k)

            results = []
            for doc, distance_score in docs:
                # Convert distance to similarity (higher is better)
                similarity_score = max(0, 1 - distance_score)

                # Apply minimum threshold
                if similarity_score > 0.1:  # Only include reasonably relevant results
                    result = RetrievalResult(
                        content=doc.page_content,
                        metadata=doc.metadata,
                        score=float(similarity_score),
                        source_table=doc.metadata.get("source", "unknown"),
                        retrieval_method="semantic",
                        relevance_explanation="Semantic similarity match",
                    )
                    results.append(result)

            return results

        except Exception as e:
            logger.error(f"Semantic search failed: {e}")
            return []

    def explain_retrieval(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        """Explain the retrieval strategy that would be used for a query"""
        query_analysis = self.query_analyzer.analyze_query(query)
        
        # Determine strategy
        if query_analysis.query_type == "aggregation":
            strategy = "aggregation"
            explanation = "Query contains aggregation keywords, using LLM-based aggregation with table selection"
        else:
            strategy = self._choose_strategy(query_analysis)
            if strategy == "semantic":
                explanation = "High confidence query, using semantic similarity search"
            elif strategy == "structured":
                explanation = "Clear entities detected, using structured entity/keyword search"
            else:
                explanation = "Mixed signals, using hybrid approach combining semantic and structured search"
        
        # Get table recommendations
        table_recommendations = []
        if hasattr(self, 'table_schemas'):
            # Simple keyword matching for table recommendations
            query_lower = query.lower()
            for table_name, info in self.table_schemas.items():
                table_desc = info.get('description', '').lower()
                table_display = info.get('display_name', table_name).lower()
                
                if any(keyword in table_desc or keyword in table_display 
                    for keyword in query_analysis.keywords):
                    table_recommendations.append(table_name)
        
        return {
            "strategy_chosen": strategy,
            "strategy_explanation": explanation,
            "analysis": {
                "query_type": query_analysis.query_type,
                "confidence": query_analysis.confidence,
                "entities_found": query_analysis.entities,
                "keywords_found": query_analysis.keywords,
                "potential_columns": query_analysis.potential_columns
            },
            "table_recommendations": table_recommendations[:3]  # Top 3
        }

    # 5. Add missing search_by_table method:
    def search_by_table(self, query: str, table_name: str, top_k: int = 5, display_mode: str = "smart") -> List[RetrievalResult]:
        """Search within a specific table"""
        try:
            # Get all documents from the specific table
            all_docs = self.vector_store.similarity_search("", k=1000)
            table_docs = [doc for doc in all_docs if doc.metadata.get("source") == table_name]
            
            if not table_docs:
                logger.warning(f"No documents found for table: {table_name}")
                return []
            
            # Search within table documents using semantic similarity
            # This is a simplified approach - in production you might want to create a sub-index
            query_analysis = self.query_analyzer.analyze_query(query)
            
            # Score documents based on query relevance
            results = []
            for doc in table_docs[:top_k * 2]:  # Get more docs to score and filter
                # Simple scoring based on keyword/entity presence
                content_lower = doc.page_content.lower()
                query_lower = query.lower()
                
                score = 0.0
                # Base score for containing query terms
                if query_lower in content_lower:
                    score += 0.5
                
                # Bonus for entity matches
                for entity in query_analysis.entities:
                    if entity.lower() in content_lower:
                        score += 0.3
                
                # Bonus for keyword matches
                for keyword in query_analysis.keywords:
                    if keyword in content_lower:
                        score += 0.1
                
                if score > 0:
                    result = RetrievalResult(
                        content=doc.page_content,
                        metadata=doc.metadata,
                        score=min(1.0, score),
                        source_table=table_name,
                        retrieval_method="table_filtered_search",
                        relevance_explanation=f"Table-specific search in {table_name}"
                    )
                    results.append(result)
            
            # Sort by score and return top results
            results.sort(key=lambda x: x.score, reverse=True)
            return self._post_process_results(results[:top_k], query_analysis, top_k, display_mode)
            
        except Exception as e:
            logger.error(f"Table search failed for {table_name}: {e}")
            return []

    def retrieve_with_full_columns(self, query: str, top_k: int = 5) -> List[RetrievalResult]:
        """Retrieve results showing all columns"""
        return self.retrieve(query, top_k, retrieval_strategy="auto", display_mode="full")

    def show_help(self):
        """Show help information"""
        help_text = """
        Available commands:
        - Enter a query to search across all tables
        - 'tables' - Show available tables and their column counts  
        - 'explain <query>' - Analyze query and show retrieval strategy
        - 'table <table_name> <query>' - Search within a specific table
        - 'full <query>' - Show all columns in results (not truncated)
        - 'help' - Show this help message
        - 'samples' - Show sample queries you can try
        - 'quit' or 'exit' - Exit the program
        
        Query Types Supported:
        - Specific searches: "show cartons with taxes foncieres"
        - Status: "show current status of cartons with taxes foncieres"
        - News: "latest news about document archiving today"
        - General: "what is document management in general"
        - Aggregation: "count cartons per user"
        """
        print(help_text)

    def get_sample_queries(self) -> List[str]:
        """Get sample queries for testing"""
        return [
            "show cartons with taxes foncieres",
            "show current status of cartons with taxes foncieres",
            "latest news about document archiving today",
            "what is document management in general",
            "count cartons per user"
        ]
    
    
    
    
    def _get_table_metadata(self, table_name: str) -> Dict[str, Any]:
        """Get metadata for a table name"""
        if not hasattr(self, 'metadata_definitions'):
            return {
                "display_name": table_name,
                "description": f"Data table: {table_name}",
                "languages": {"en": "", "fr": ""}
            }
        
        metadata = self.metadata_definitions.get("table_names", {}).get(table_name, {})
        
        display_parts = [table_name]
        if metadata.get("en"):
            display_parts.append(f"({metadata['en']})")
        if metadata.get("fr") and metadata["fr"] != metadata.get("en"):
            display_parts.append(f"({metadata['fr']})")
        
        return {
            "display_name": " ".join(display_parts),
            "description": metadata.get("description", f"Data table: {table_name}"),
            "languages": {"en": metadata.get("en", ""), "fr": metadata.get("fr", "")}
        }
    
    def _get_column_metadata(self, column_name: str) -> Dict[str, Any]:
        """Get metadata for a column name"""
        metadata = self.metadata_definitions.get("column_names", {}).get(column_name, {})
        
        display_parts = [column_name]
        if metadata.get("en"):
            display_parts.append(f"({metadata['en']})")
        if metadata.get("fr") and metadata["fr"] != metadata.get("en"):
            display_parts.append(f"({metadata['fr']})")
        
        return {
            "display_name": " ".join(display_parts),
            "description": metadata.get("description", f"Column: {column_name}"),
            "languages": {"en": metadata.get("en", ""), "fr": metadata.get("fr", "")}
        }
        
    def get_table_info(self) -> Dict[str, Dict]:
        """Get information about available tables"""
        return self.table_schemas.copy()

    # 3. Add the missing get_schema_for_llm method (REPLACE the existing one):
    def get_schema_for_llm(self, include_samples: bool = False) -> str:
        """Return enhanced schema with metadata for LLM prompts"""
        schema = {}
        for table_name, info in self.table_schemas.items():
            # Enhanced table info with metadata
            schema[table_name] = {
                "display_name": info.get("display_name", table_name),
                "description": info.get("description", ""),
                "columns": []
            }
            
            # Enhanced column info with metadata
            for col in info.get("columns", []):
                col_info = {
                    "name": col,
                    "display_name": info.get("column_metadata", {}).get(col, {}).get("display_name", col),
                    "description": info.get("column_metadata", {}).get(col, {}).get("description", "")
                }
                
                if include_samples:
                    col_info["sample_values"] = info.get("sample_values", {}).get(col, [])
                
                schema[table_name]["columns"].append(col_info)
        
        return json.dumps(schema, indent=2, ensure_ascii=False)

    def _select_relevant_tables_for_query(self, query: str, query_analysis: QueryAnalysis) -> List[str]:
        """Select most relevant tables for aggregation query using LLM"""
        schema_str = self.get_schema_for_llm(include_samples=False)
        
        selection_prompt = f"""
        Given this database schema with table and column descriptions:
        {schema_str}
        
        For the business query: "{query}"
        
        Identify which tables are most relevant. Consider:
        1. Table descriptions and purposes
        2. Column names and their meanings (both original and translated names)
        3. Query entities: {query_analysis.entities}
        4. Query keywords: {query_analysis.keywords}
        
        Return ONLY a Python list of relevant table names, like: ["table1", "table2"]
        """
        
        try:
            response = self.llm_client.chat(
                message=selection_prompt,
                model=self.llm_model,
                temperature=0.1,  # Low temperature for consistent selection
                max_tokens=512
            )
            
            # Parse the response to extract table list
            response_text = response.text.strip()
            
            # Try to extract list from response
            list_match = re.search(r'\[([^\]]+)\]', response_text)
            if list_match:
                list_content = list_match.group(1)
                # Clean and parse
                tables = [t.strip().strip('"\'') for t in list_content.split(',')]
                # Validate tables exist
                valid_tables = [t for t in tables if t in self.table_schemas]
                logger.info(f"LLM selected tables: {valid_tables}")
                return valid_tables
            
            return list(self.table_schemas.keys())[:3]  # Fallback
            
        except Exception as e:
            logger.error(f"Table selection failed: {e}")
            return list(self.table_schemas.keys())[:3]
      
    def _load_dataframes_from_tables(self, relevant_tables: List[str]) -> Dict[str, pd.DataFrame]:
        """Load CSV files directly into pandas DataFrames"""
        dataframes = {}
        
        for table_name in relevant_tables:
            csv_path = self._find_csv_file(table_name)
            
            if csv_path and os.path.exists(csv_path):
                try:
                    df = pd.read_csv(csv_path)
                    clean_table_name = Path(table_name).stem
                    dataframes[clean_table_name] = df
                    logger.info(f"Loaded DataFrame '{clean_table_name}' with {len(df)} rows from {csv_path}")
                except Exception as e:
                    logger.error(f"Failed to load DataFrame {table_name}: {e}")
        
        return dataframes
    
    def _find_csv_file(self, table_name: str) -> str:
        """Find the CSV file path for a given table name in data/documents"""
        
        # Remove .csv extension if present
        clean_table_name = table_name.replace('.csv', '')
        
        # Fix the path construction - use os.path.join for proper path handling
        data_dir = r"C:\Users\jguesmi\Downloads\conversation-RAG-main\conversation-RAG-main\data"
        documents_dir = "documents"
        
        # Try multiple possible paths
        possible_paths = [
            os.path.join(data_dir, documents_dir, f"{clean_table_name}.csv"),
            os.path.join(data_dir, documents_dir, f"{clean_table_name}"),
            f"{clean_table_name}.csv"
        ]
        
        for csv_path in possible_paths:
            if os.path.exists(csv_path):
                logger.info(f"Found CSV file for table '{table_name}' at: {csv_path}")
                return csv_path
        
        # If not found, try to search in current directory and subdirectories
        for root, dirs, files in os.walk("."):
            for file in files:
                if file == f"{clean_table_name}.csv":
                    found_path = os.path.join(root, file)
                    logger.info(f"Found CSV file for table '{table_name}' at: {found_path}")
                    return found_path
    
        logger.error(f"CSV file not found for table: {clean_table_name}")
        return None

    def _generate_sql_for_aggregation(self, query: str, query_analysis: QueryAnalysis, 
                                    relevant_tables: List[str]) -> SQLGenerationResult:
        """Generate SQL query for aggregation using LLM"""
        
        # Create focused schema for SQL generation
        focused_schema = {}
        for table_name in relevant_tables:
            if table_name in self.table_schemas:
                info = self.table_schemas[table_name]
                focused_schema[table_name] = {
                    "display_name": info.get("display_name", table_name),
                    "description": info.get("description", ""),
                    "columns": []
                }
                
                for col in info.get("columns", []):
                    col_metadata = info.get("column_metadata", {}).get(col, {})
                    focused_schema[table_name]["columns"].append({
                        "name": col,
                        "display_name": col_metadata.get("display_name", col),
                        "sample_values": info.get("sample_values", {}).get(col, [])[:3]
                    })
        
        schema_str = json.dumps(focused_schema, indent=2, ensure_ascii=False)
        
        # Enhanced SQL generation prompt
        sql_prompt = f"""
        You are a SQL expert. Generate a SQLite query for this business question.
        
        Database Schema:
        {schema_str}
        
        Business Query: "{query}"
        Detected entities: {query_analysis.entities}
        Detected keywords: {query_analysis.keywords}
        
        Requirements:
        1. Generate ONLY valid SQLite syntax
        2. Use original column names (like 'nom', 'prenom', not translations)
        3. Handle aggregations: COUNT, SUM, AVG, GROUP BY, etc.
        4. Include appropriate WHERE clauses for filtering
        5. Use proper JOINs if multiple tables needed
        6. Return results in logical order (ORDER BY)
        
        Return as JSON:
        {{
            "sql": "SELECT ... FROM ... WHERE ... GROUP BY ... ORDER BY ...",
            "explanation": "Brief explanation of what this query does",
            "tables_used": ["table1", "table2"],
            "columns_used": ["table1.column1", "table2.column2"],
            "aggregation_type": "count|sum|avg|group_by|filter"
        }}
        """
        
        try:
            response = self.llm_client.chat(
                message=sql_prompt,
                model=self.llm_model,
                temperature=0.1,  # Low temperature for consistent SQL generation
                max_tokens=1024
            )
            
            # Parse response
            response_text = response.text.strip()
            
            # Extract JSON from response
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                result_dict = json.loads(json_str)
                
                return SQLGenerationResult(
                    sql_query=result_dict.get("sql", ""),
                    explanation=result_dict.get("explanation", ""),
                    tables_used=result_dict.get("tables_used", []),
                    columns_used=result_dict.get("columns_used", []),
                    aggregation_type=result_dict.get("aggregation_type", "unknown"),
                    confidence=0.8
                )
            else:
                # Fallback parsing
                return SQLGenerationResult(
                    sql_query=response_text,
                    explanation="Generated SQL query",
                    tables_used=relevant_tables,
                    columns_used=[],
                    aggregation_type="unknown",
                    confidence=0.5
                )
                
        except Exception as e:
            logger.error(f"SQL generation failed: {e}")
            return SQLGenerationResult(
                sql_query="",
                explanation=f"Error generating SQL: {str(e)}",
                tables_used=[],
                columns_used=[],
                aggregation_type="error",
                confidence=0.0,
                errors=[str(e)]
            )
            
    def _enhanced_aggregation_search(self, query: str, query_analysis: QueryAnalysis, top_k: int) -> List[RetrievalResult]:
        """Enhanced aggregation retrieval with SQL generation"""
        self._analyze_table_schemas(use_metadata=True)
        # Step 1: Select relevant tables
        relevant_tables = self._select_relevant_tables_for_query(query, query_analysis)
        logger.info(f"Selected relevant tables for SQL: {relevant_tables}")
        
        # Step 2: Get sample data from relevant tables
        retrieval_results = []
        for table_name in relevant_tables:
            # Get more data for better SQL context
            table_results = self.search_by_table(query, table_name, top_k=20, display_mode="full")
            retrieval_results.extend(table_results)
        
        # Step 3: Generate SQL query
        sql_result = self._generate_sql_for_aggregation(query, query_analysis, relevant_tables)
        
        dataframes = self._load_dataframes_from_tables(relevant_tables)
        
        # Step 5: Return result with SQL and database path
        return [RetrievalResult(
            content=f"SQL Query: {sql_result.sql_query}\nExplanation: {sql_result.explanation}",
            metadata={
                "sql_generation_result": sql_result,
                "dataframes": dataframes,
                "retrieval_results": retrieval_results,
                "relevant_tables": relevant_tables
            },
            score=sql_result.confidence,
            source_table="_sql_aggregation_",
            retrieval_method="sql_generation",
            relevance_explanation=sql_result.explanation,
            formatted_content=f"Tables: {sql_result.tables_used}\nAggregation: {sql_result.aggregation_type}\nSQL: {sql_result.sql_query}"
        )]
          
    def _aggregation_search(self, query: str, query_analysis: QueryAnalysis, top_k: int) -> List[RetrievalResult]:
        """Use enhanced SQL-based aggregation instead of pandas"""
        return self._enhanced_aggregation_search(query, query_analysis, top_k)

    def _parse_llm_aggregation_response(self, response: str) -> Dict[str, Any]:
        """Parse LLM response for aggregation"""
        try:
            # Try to find dictionary in response
            dict_match = re.search(r'\{.*\}', response, re.DOTALL)
            if dict_match:
                dict_str = dict_match.group(0)
                # Clean up the dictionary string
                dict_str = re.sub(r'```python|```', '', dict_str)
                return ast.literal_eval(dict_str)
            
            # Fallback parsing
            result = {}
            
            # Extract tables
            tables_match = re.search(r'"relevant_tables":\s*\[(.*?)\]', response)
            if tables_match:
                result['relevant_tables'] = [t.strip().strip('"\'') for t in tables_match.group(1).split(',')]
            
            # Extract columns  
            cols_match = re.search(r'"columns_needed":\s*\[(.*?)\]', response)
            if cols_match:
                result['columns_needed'] = [c.strip().strip('"\'') for c in cols_match.group(1).split(',')]
            
            # Extract code
            code_match = re.search(r'"code":\s*"(.*?)"', response, re.DOTALL)
            if code_match:
                result['code'] = code_match.group(1).replace('\\n', '\n')
            
            # Extract explanation
            exp_match = re.search(r'"explanation":\s*"(.*?)"', response)
            if exp_match:
                result['explanation'] = exp_match.group(1)
                
            return result
            
        except Exception as e:
            logger.error(f"Failed to parse LLM response: {e}")
            return {
                "relevant_tables": [],
                "columns_needed": [],
                "code": response,
                "explanation": "Failed to parse LLM response"
            }
    
    def process_query(self, query_request: QueryRequest) -> Dict[str, Any]:
        """Main entry point for processing queries with full configuration"""
        
        try:
            # Validate input
            if not query_request.query.strip():
                return {
                    "success": False,
                    "error": "Empty query provided",
                    "results": []
                }
            
            # Analyze the query
            query_analysis = self.query_analyzer.analyze_query(query_request.query)
            
            # Get retrieval results
            if query_request.table_filter:
                results = self.search_by_table(
                    query_request.query, 
                    query_request.table_filter, 
                    query_request.top_k,
                    query_request.display_mode
                )
            else:
                results = self.retrieve(
                    query_request.query,
                    query_request.top_k,
                    query_request.retrieval_strategy,
                    query_request.display_mode
                )
            
            # Filter by minimum score
            results = [r for r in results if r.score >= query_request.min_score]
            
            # Prepare response
            response = {
                "success": True,
                "query": query_request.query,
                "total_results": len(results),
                "results": []
            }
            
            # Format results
            for result in results:
                result_dict = {
                    "content": result.formatted_content if hasattr(result, 'formatted_content') and result.formatted_content else result.content,
                    "score": result.score,
                    "source_table": result.source_table,
                    "retrieval_method": result.retrieval_method
                }
                
                if query_request.include_metadata:
                    result_dict["metadata"] = result.metadata
                    result_dict["relevance_explanation"] = result.relevance_explanation
                
                response["results"].append(result_dict)
            
            # Add explanation if requested
            if query_request.explain_results:
                response["explanation"] = self.explain_retrieval(query_request.query, query_request.top_k)
                response["query_analysis"] = {
                    "query_type": query_analysis.query_type,
                    "confidence": query_analysis.confidence,
                    "entities": query_analysis.entities,
                    "keywords": query_analysis.keywords,
                    "potential_columns": query_analysis.potential_columns
                }
            
            return response
        except Exception as e:
            logger.error(f"Query processing failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "results": []
            }


    def simple_query(self, query: str, **kwargs) -> List[Dict[str, Any]]:
        """Simplified query interface that returns just the results"""
        
        query_request = QueryRequest(
            query=query,
            top_k=kwargs.get('top_k', 5),
            retrieval_strategy=kwargs.get('strategy', 'auto'),
            display_mode=kwargs.get('display_mode', 'smart'),
            table_filter=kwargs.get('table', None),
            min_score=kwargs.get('min_score', 0.1)
        )
        
        response = self.process_query(query_request)
        return response.get('results', [])


    def interactive_query(self) -> None:
        """Interactive query interface for testing"""
        
        print("=== Interactive RAG Query Interface ===")
        print("Type your queries below. Commands:")
        print("  'tables' - Show available tables")
        print("  'explain <query>' - Explain retrieval strategy")
        print("  'quit' or 'exit' - Exit")
        print("=" * 50)
        
        while True:
            try:
                user_input = input("\nQuery: ").strip()
                
                if not user_input:
                    continue
                    
                if user_input.lower() in ['quit', 'exit']:
                    print("Goodbye!")
                    break
                
                if user_input.lower() == 'tables':
                    tables = self.get_table_info()
                    print(f"\nAvailable tables ({len(tables)}):")
                    for table_name, info in tables.items():
                        print(f"  - {table_name}: {len(info['columns'])} columns")
                    continue
                
                if user_input.lower().startswith('explain '):
                    query = user_input[8:].strip()
                    if query:
                        explanation = self.explain_retrieval(query)
                        print(f"\nQuery Analysis for: '{query}'")
                        print(f"  Strategy: {explanation['strategy_chosen']}")
                        print(f"  Confidence: {explanation['analysis']['confidence']:.2f}")
                        print(f"  Entities: {explanation['analysis']['entities_found']}")
                        print(f"  Keywords: {explanation['analysis']['keywords_found']}")
                    continue
                
                # Process regular query
                results = self.simple_query(user_input, top_k=3)
                
                if not results:
                    print("No results found.")
                    continue
                
                print(f"\nFound {len(results)} results:")
                for i, result in enumerate(results, 1):
                    print(f"\n{i}. [{result['source_table']}] Score: {result['score']:.3f}")
                    print(f"   Content: {result['content'][:200]}...")
                    print(f"   Method: {result['retrieval_method']}")
                    
            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except Exception as e:
                print(f"Error: {e}")


    def batch_query(self, queries: List[str], **kwargs) -> Dict[str, List[Dict[str, Any]]]:
        """Process multiple queries at once"""
        
        results = {}
        
        for query in queries:
            try:
                query_results = self.simple_query(query, **kwargs)
                results[query] = query_results
            except Exception as e:
                logger.error(f"Batch query failed for '{query}': {e}")
                results[query] = []
        
        return results


    def query_with_filters(self, query: str, filters: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Query with advanced filtering options"""
        
        # Build query request from filters
        query_request = QueryRequest(
            query=query,
            top_k=filters.get('limit', 5),
            retrieval_strategy=filters.get('strategy', 'auto'),
            display_mode=filters.get('display', 'smart'),
            table_filter=filters.get('table'),
            min_score=filters.get('min_score', 0.1),
            include_metadata=filters.get('include_metadata', True),
            explain_results=filters.get('explain', False)
        )
        
        response = self.process_query(query_request)
        
        # Apply additional filters
        results = response.get('results', [])
        
        # Filter by source table (if multiple specified)
        if 'tables' in filters and isinstance(filters['tables'], list):
            results = [r for r in results if r['source_table'] in filters['tables']]
        
        # Filter by score range
        if 'score_range' in filters:
            min_s, max_s = filters['score_range']
            results = [r for r in results if min_s <= r['score'] <= max_s]
        
        # Sort by different criteria
        if 'sort_by' in filters:
            reverse = filters.get('sort_desc', True)
            if filters['sort_by'] == 'score':
                results.sort(key=lambda x: x['score'], reverse=reverse)
            elif filters['sort_by'] == 'table':
                results.sort(key=lambda x: x['source_table'], reverse=reverse)
        
        return results[:filters.get('limit', len(results))]






# Usage example and testing functions
def test_retrieval_system():
    """Interactive test of the retrieval system"""
    print("Interactive RAG Retrieval System")
    print("=" * 50)

    try:
        # Initialize retriever
        retriever = HybridRetriever()

        if not retriever.vector_store:
            print("Vector store not available. Please ensure embeddings are created first.")
            return

        # Show available tables
        print(f"Available tables: {list(retriever.get_table_info().keys())}")
        print("\nCommands:")
        print("  - Enter a query to search")
        print("  - 'tables' to show table info")
        print("  - 'explain <query>' to see retrieval strategy")
        print("  - 'table <table_name> <query>' to search specific table")
        print("  - 'full <query>' to show all columns")
        print("  - 'quit' to exit")
        print("-" * 50)

        while True:
            try:
                user_input = input("\nEnter query (or command): ").strip()
                
                if not user_input:
                    continue
                    
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("Goodbye!")
                    break
                
                if user_input.lower() == 'tables':
                    print("\nTable Information:")
                    for table_name, info in retriever.get_table_info().items():
                        print(f"\n  {table_name}:")
                        print(f"    Columns: {', '.join(info['columns'][:10])}{'...' if len(info['columns']) > 10 else ''}")
                        print(f"    Total columns: {len(info['columns'])}")
                    continue
                
                if user_input.lower().startswith('explain '):
                    query = user_input[8:].strip()
                    if query:
                        explanation = retriever.explain_retrieval(query)
                        print(f"\nQuery Analysis for: '{query}'")
                        print(f"  Strategy: {explanation['strategy_chosen']}")
                        print(f"  Confidence: {explanation['analysis']['confidence']:.2f}")
                        print(f"  Query type: {explanation['analysis']['query_type']}")
                        print(f"  Entities: {explanation['analysis']['entities_found']}")
                        print(f"  Keywords: {explanation['analysis']['keywords_found']}")
                        print(f"  Potential columns: {explanation['analysis']['potential_columns']}")
                        print(f"  Explanation: {explanation['strategy_explanation']}")
                        if explanation['table_recommendations']:
                            print(f"  Recommended tables: {explanation['table_recommendations']}")
                    continue
                
                if user_input.lower().startswith('table '):
                    parts = user_input[6:].strip().split(' ', 1)
                    if len(parts) == 2:
                        table_name, query = parts
                        print(f"\nSearching in table '{table_name}' for: '{query}'")
                        results = retriever.search_by_table(query, table_name, top_k=5)
                        _display_results(results, query, show_all_columns=True)
                    else:
                        print("Usage: table <table_name> <query>")
                    continue
                
                if user_input.lower().startswith('full '):
                    query = user_input[5:].strip()
                    if query:
                        print(f"\nSearching with full columns for: '{query}'")
                        results = retriever.retrieve_with_full_columns(query, top_k=5)
                        _display_results(results, query, show_all_columns=True)
                    continue
                if user_input.lower() == 'help':
                        retriever.show_help()
                        continue
                
                if user_input.lower() == 'samples':
                    print("\nSample queries to try:")
                    for i, sample in enumerate(retriever.get_sample_queries(), 1):
                        print(f"  {i}. {sample}")
                    continue        
                # Regular search
                query = user_input
                print(f"\nSearching for: '{query}'")
                
                # Show strategy explanation
                explanation = retriever.explain_retrieval(query)
                print(f"Using {explanation['strategy_chosen']} strategy (confidence: {explanation['analysis']['confidence']:.2f})")
                
                # Perform search
                results = retriever.retrieve(query, top_k=5, display_mode="smart")
                _display_results(results, query)
                
            except KeyboardInterrupt:
                print("\n\nGoodbye!")
                break
            except Exception as e:
                print(f"Error processing query: {e}")
                logger.error(f"Query processing error: {e}")

    except Exception as e:
        logger.error(f"Test initialization failed: {e}")
        print(f"Test failed: {e}")


def _display_results(results: List[RetrievalResult], query: str, show_all_columns: bool = False):
    """Helper function to display search results"""
    if not results:
        print("  No results found.")
        return
    
    print(f"  Found {len(results)} results:")
    print("-" * 80)
    
    for i, result in enumerate(results, 1):
        print(f"  {i}. [{result.source_table}] Score: {result.score:.3f}")
        
        # Choose content to display
        if show_all_columns:
            display_content = result.content
        else:
            display_content = (
                result.formatted_content 
                if hasattr(result, 'formatted_content') and result.formatted_content 
                else result.content
            )
        
        # Truncate very long content for readability
        if len(display_content) > 500:
            display_content = display_content[:500] + "..."
        
        print(f"     Content: {display_content}")
        print(f"     Method: {result.retrieval_method}")
        
        if result.relevance_explanation:
            print(f"     Relevance: {result.relevance_explanation}")
        
        print("-" * 40)

def interactive_query_session():
    """Start an interactive query session"""
    test_retrieval_system()
    

if __name__ == "__main__":
    interactive_query_session()
