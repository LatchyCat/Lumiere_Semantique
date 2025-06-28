# backend/lumiere_core/services/oracle_service.py

import logging
import json
import networkx as nx
from typing import Dict, Any, List, Optional, Tuple, Set, Union
from collections import defaultdict, Counter
import re
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError
import hashlib
import time
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
from functools import lru_cache
import threading
from queue import Queue, Empty
import asyncio
from contextlib import contextmanager

from . import llm_service, cortex_service, ollama
from .llm_service import TaskType
from .utils import clean_llm_code_output

logger = logging.getLogger(__name__)

# --- Exceptions ---
class OracleServiceError(Exception):
    """Base exception for Oracle service errors."""
    pass

class GraphBuildingError(OracleServiceError):
    """Raised when graph building fails."""
    pass

class EntityExtractionError(OracleServiceError):
    """Raised when entity extraction fails."""
    pass

class GraphAnalysisError(OracleServiceError):
    """Raised when graph analysis fails."""
    pass

class RAGSearchError(OracleServiceError):
    """Raised when RAG search fails."""
    pass

# --- Enhanced Configuration ---
class SearchStrategy(Enum):
    """Search strategies for different query types."""
    SIMPLE = "simple"
    EXPANDED = "expanded"
    MULTI_HOP = "multi_hop"
    HYBRID = "hybrid"
    GRAPH_GUIDED = "graph_guided"

@dataclass
class RAGConfig:
    """Enhanced configuration for RAG search behavior."""
    # Search parameters
    default_k: int = 20
    max_k: int = 50
    min_k: int = 5

    # Reranking thresholds
    similarity_threshold: float = 0.7
    diversity_threshold: float = 0.3
    relevance_decay: float = 0.95

    # Query expansion
    max_expansions: int = 5
    max_synonyms: int = 3
    expansion_depth: int = 2

    # Multi-hop settings
    max_hops: int = 3
    hop_decay: float = 0.8
    min_hop_relevance: float = 0.5

    # Parallel execution
    max_workers: int = 5
    timeout_seconds: int = 30

    # Caching
    cache_ttl: int = 3600  # 1 hour
    max_cache_size: int = 1000

    # Advanced features
    use_semantic_clustering: bool = True
    use_entity_linking: bool = True
    use_code_understanding: bool = True
    enable_self_reflection: bool = True

# Global config instance
rag_config = RAGConfig()

# --- Cache Management ---
class SearchCache:
    """Thread-safe cache for search results."""
    def __init__(self, max_size: int = 1000, ttl: int = 3600):
        self._cache = {}
        self._timestamps = {}
        self._lock = threading.RLock()
        self._max_size = max_size
        self._ttl = ttl

    def get(self, key: str) -> Optional[Any]:
        with self._lock:
            if key in self._cache:
                if time.time() - self._timestamps[key] < self._ttl:
                    return self._cache[key]
                else:
                    del self._cache[key]
                    del self._timestamps[key]
            return None

    def put(self, key: str, value: Any):
        with self._lock:
            if len(self._cache) >= self._max_size:
                # Remove oldest entry
                oldest_key = min(self._timestamps, key=self._timestamps.get)
                del self._cache[oldest_key]
                del self._timestamps[oldest_key]

            self._cache[key] = value
            self._timestamps[key] = time.time()

    def clear(self):
        with self._lock:
            self._cache.clear()
            self._timestamps.clear()

# Global cache instance
search_cache = SearchCache(rag_config.max_cache_size, rag_config.cache_ttl)

# --- Advanced Entity Processing ---
@dataclass
class Entity:
    """Represents an extracted entity with metadata."""
    text: str
    type: str
    confidence: float = 1.0
    context: str = ""
    aliases: List[str] = field(default_factory=list)
    related: List[str] = field(default_factory=list)

class EntityExtractor:
    """Advanced entity extraction with multiple strategies."""

    def __init__(self, config: RAGConfig):
        self.config = config
        self._pattern_cache = {}
        self._init_patterns()

    def _init_patterns(self):
        """Initialize regex patterns for code entity extraction."""
        self._pattern_cache = {
            'import': re.compile(r'(?:from|import)\s+([a-zA-Z_][\w\.]*)', re.MULTILINE),
            'class': re.compile(r'class\s+([A-Z]\w*)', re.MULTILINE),
            'function': re.compile(r'def\s+([a-z_]\w*)', re.MULTILINE),
            'variable': re.compile(r'([a-z_]\w*)\s*=', re.MULTILINE),
            'file_path': re.compile(r'[\'"]([^\'"\s]+\.[a-z]+)[\'"]', re.MULTILINE),
            'url': re.compile(r'https?://[^\s]+', re.MULTILINE),
            'api_endpoint': re.compile(r'[\'"][/][\w/\-{}]+[\'"]', re.MULTILINE),
        }

    def extract(self, text: str, use_llm: bool = True) -> Dict[str, List[Entity]]:
        """Extract entities using both pattern matching and LLM."""
        entities = defaultdict(list)

        # Pattern-based extraction
        if self.config.use_code_understanding:
            entities.update(self._extract_with_patterns(text))

        # LLM-based extraction
        if use_llm:
            try:
                llm_entities = self._extract_with_llm(text)
                self._merge_entities(entities, llm_entities)
            except Exception as e:
                logger.warning(f"LLM entity extraction failed: {e}")

        # Entity linking and expansion
        if self.config.use_entity_linking:
            entities = self._link_entities(entities)

        return dict(entities)

    def _extract_with_patterns(self, text: str) -> Dict[str, List[Entity]]:
        """Extract entities using regex patterns."""
        entities = defaultdict(list)

        for pattern_name, pattern in self._pattern_cache.items():
            matches = pattern.findall(text)
            entity_type = self._pattern_to_entity_type(pattern_name)

            for match in matches:
                if isinstance(match, str) and match.strip():
                    entity = Entity(
                        text=match.strip(),
                        type=entity_type,
                        confidence=0.8,
                        context=self._extract_context(text, match)
                    )
                    entities[entity_type].append(entity)

        return entities

    def _extract_with_llm(self, text: str) -> Dict[str, List[Entity]]:
        """Extract entities using LLM with structured output."""
        prompt = f"""Extract software development entities from this text.
Return a JSON object with these keys: functions, classes, files, technologies, patterns, concepts.
Each value should be a list of objects with: text, confidence (0-1), and context.

Text: "{text[:1000]}"

JSON Output:"""

        try:
            response = llm_service.generate_text(prompt, task_type=TaskType.SIMPLE)
            cleaned = clean_llm_code_output(response)
            data = json.loads(cleaned)

            entities = defaultdict(list)
            for entity_type, entity_list in data.items():
                if isinstance(entity_list, list):
                    for item in entity_list:
                        if isinstance(item, dict) and 'text' in item:
                            entity = Entity(
                                text=item['text'],
                                type=entity_type,
                                confidence=item.get('confidence', 0.9),
                                context=item.get('context', '')
                            )
                            entities[entity_type].append(entity)

            return entities
        except Exception as e:
            logger.error(f"Failed to parse LLM entity response: {e}")
            return {}

    def _merge_entities(self, base: Dict[str, List[Entity]],
                       new: Dict[str, List[Entity]]) -> None:
        """Merge entity lists, avoiding duplicates."""
        for entity_type, entity_list in new.items():
            existing_texts = {e.text.lower() for e in base.get(entity_type, [])}

            for entity in entity_list:
                if entity.text.lower() not in existing_texts:
                    base[entity_type].append(entity)
                    existing_texts.add(entity.text.lower())

    def _link_entities(self, entities: Dict[str, List[Entity]]) -> Dict[str, List[Entity]]:
        """Link related entities and find aliases."""
        # Build entity graph
        entity_graph = defaultdict(set)

        for entity_type, entity_list in entities.items():
            for entity in entity_list:
                # Find potential aliases (e.g., MyClass vs my_class)
                aliases = self._find_aliases(entity.text, entities)
                entity.aliases = aliases

                # Link related entities
                for alias in aliases:
                    entity_graph[entity.text].add(alias)

        return entities

    def _find_aliases(self, text: str, all_entities: Dict[str, List[Entity]]) -> List[str]:
        """Find potential aliases for an entity."""
        aliases = []
        text_lower = text.lower()
        text_snake = self._to_snake_case(text)
        text_camel = self._to_camel_case(text)

        for entity_list in all_entities.values():
            for entity in entity_list:
                if entity.text != text:
                    other_lower = entity.text.lower()
                    if (other_lower == text_lower or
                        other_lower == text_snake or
                        other_lower == text_camel):
                        aliases.append(entity.text)

        return aliases

    def _extract_context(self, text: str, match: str, window: int = 50) -> str:
        """Extract context around a match."""
        index = text.find(match)
        if index == -1:
            return ""

        start = max(0, index - window)
        end = min(len(text), index + len(match) + window)
        return text[start:end].strip()

    def _pattern_to_entity_type(self, pattern_name: str) -> str:
        """Map pattern names to entity types."""
        mapping = {
            'import': 'technologies',
            'class': 'classes',
            'function': 'functions',
            'variable': 'variables',
            'file_path': 'files',
            'url': 'urls',
            'api_endpoint': 'endpoints'
        }
        return mapping.get(pattern_name, 'general_terms')

    def _to_snake_case(self, text: str) -> str:
        """Convert to snake_case."""
        s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', text)
        return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()

    def _to_camel_case(self, text: str) -> str:
        """Convert to CamelCase."""
        components = text.split('_')
        return ''.join(x.title() for x in components)

# --- Query Processing and Expansion ---
class QueryProcessor:
    """Advanced query processing with multiple expansion strategies."""

    def __init__(self, config: RAGConfig):
        self.config = config
        self._synonym_cache = {}

    def process(self, query: str, entities: Dict[str, List[Entity]],
                strategy: SearchStrategy = SearchStrategy.EXPANDED) -> List[str]:
        """Process and expand query based on strategy."""
        if strategy == SearchStrategy.SIMPLE:
            return [query]
        elif strategy == SearchStrategy.EXPANDED:
            return self._expand_query(query, entities)
        elif strategy == SearchStrategy.MULTI_HOP:
            return self._generate_hop_queries(query, entities)
        elif strategy == SearchStrategy.HYBRID:
            return self._hybrid_expansion(query, entities)
        else:
            return [query]

    def _expand_query(self, query: str, entities: Dict[str, List[Entity]]) -> List[str]:
        """Expand query with entity variations and synonyms."""
        expansions = [query]

        # Add entity-based expansions
        for entity_type, entity_list in entities.items():
            for entity in entity_list[:2]:  # Limit to avoid explosion
                # Add direct entity queries
                expansions.append(f"{entity.text} implementation")
                expansions.append(f"usage of {entity.text}")

                # Add alias queries
                for alias in entity.aliases[:2]:
                    expansions.append(f"{alias} {entity_type[:-1]}")

        # Add pattern-based expansions
        patterns = self._extract_query_patterns(query)
        for pattern in patterns:
            expansions.extend(self._expand_pattern(pattern, entities))

        # Deduplicate and limit
        seen = set()
        unique = []
        for exp in expansions:
            if exp not in seen and len(exp) > 3:
                seen.add(exp)
                unique.append(exp)

        return unique[:self.config.max_expansions]

    def _generate_hop_queries(self, query: str, entities: Dict[str, List[Entity]],
                             hop: int = 0) -> List[str]:
        """Generate queries for multi-hop search."""
        if hop == 0:
            return [query]

        hop_queries = []

        # Focus on different aspects based on hop
        if hop == 1:
            # Look for implementations
            for entity in self._get_top_entities(entities, 3):
                hop_queries.append(f"implementation of {entity.text}")
                hop_queries.append(f"{entity.text} definition")
        elif hop == 2:
            # Look for usages and connections
            for entity in self._get_top_entities(entities, 2):
                hop_queries.append(f"calls to {entity.text}")
                hop_queries.append(f"{entity.text} dependencies")
        else:
            # Look for related concepts
            for entity in self._get_top_entities(entities, 2):
                hop_queries.append(f"related to {entity.text}")

        return hop_queries[:self.config.max_expansions]

    def _hybrid_expansion(self, query: str, entities: Dict[str, List[Entity]]) -> List[str]:
        """Combine multiple expansion strategies."""
        expansions = []

        # Simple expansions
        expansions.extend(self._expand_query(query, entities))

        # Semantic expansions
        if self.config.use_semantic_clustering:
            expansions.extend(self._semantic_expansion(query, entities))

        # Code-aware expansions
        if self.config.use_code_understanding:
            expansions.extend(self._code_aware_expansion(query, entities))

        # Deduplicate and rank
        ranked = self._rank_expansions(expansions, query)
        return ranked[:self.config.max_expansions * 2]

    def _semantic_expansion(self, query: str, entities: Dict[str, List[Entity]]) -> List[str]:
        """Generate semantically related queries."""
        expansions = []

        # Extract key concepts
        concepts = self._extract_concepts(query)

        for concept in concepts:
            # Add conceptual queries
            expansions.append(f"examples of {concept}")
            expansions.append(f"{concept} pattern")
            expansions.append(f"best practices for {concept}")

        return expansions

    def _code_aware_expansion(self, query: str, entities: Dict[str, List[Entity]]) -> List[str]:
        """Generate code-aware query expansions."""
        expansions = []

        # Look for code patterns in query
        if "api" in query.lower():
            expansions.extend(["API routes", "endpoint definitions", "REST endpoints"])
        if "database" in query.lower():
            expansions.extend(["schema", "models", "queries"])
        if "test" in query.lower():
            expansions.extend(["unit tests", "test cases", "test fixtures"])

        return expansions

    def _extract_query_patterns(self, query: str) -> List[str]:
        """Extract patterns from query."""
        patterns = []

        # Question patterns
        question_words = ["what", "how", "where", "when", "why", "which"]
        for word in question_words:
            if word in query.lower():
                patterns.append(word)

        # Action patterns
        action_words = ["implement", "use", "call", "create", "update", "delete"]
        for word in action_words:
            if word in query.lower():
                patterns.append(word)

        return patterns

    def _expand_pattern(self, pattern: str, entities: Dict[str, List[Entity]]) -> List[str]:
        """Expand based on pattern type."""
        expansions = []

        if pattern in ["what", "where"]:
            for entity in self._get_top_entities(entities, 2):
                expansions.append(f"{pattern} is {entity.text}")
        elif pattern in ["how"]:
            for entity in self._get_top_entities(entities, 2):
                expansions.append(f"{pattern} to use {entity.text}")

        return expansions

    def _extract_concepts(self, query: str) -> List[str]:
        """Extract high-level concepts from query."""
        # Simple concept extraction - could be enhanced with NLP
        words = query.lower().split()
        concepts = []

        # Look for noun phrases
        for i, word in enumerate(words):
            if len(word) > 4:  # Simple heuristic
                if i + 1 < len(words):
                    concept = f"{word} {words[i+1]}"
                    if len(words[i+1]) > 3:
                        concepts.append(concept)
                concepts.append(word)

        return concepts[:3]

    def _rank_expansions(self, expansions: List[str], original: str) -> List[str]:
        """Rank expansions by relevance to original query."""
        scored = []

        for exp in expansions:
            score = self._calculate_relevance(exp, original)
            scored.append((exp, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return [exp for exp, _ in scored]

    def _calculate_relevance(self, expansion: str, original: str) -> float:
        """Calculate relevance score between expansion and original."""
        # Simple word overlap score
        original_words = set(original.lower().split())
        expansion_words = set(expansion.lower().split())

        overlap = len(original_words & expansion_words)
        total = len(original_words | expansion_words)

        return overlap / total if total > 0 else 0

    def _get_top_entities(self, entities: Dict[str, List[Entity]], n: int) -> List[Entity]:
        """Get top N entities by confidence."""
        all_entities = []
        for entity_list in entities.values():
            all_entities.extend(entity_list)

        all_entities.sort(key=lambda e: e.confidence, reverse=True)
        return all_entities[:n]

# --- Advanced Search and Reranking ---
class SearchOrchestrator:
    """Orchestrates complex search operations with multiple strategies."""

    def __init__(self, config: RAGConfig):
        self.config = config
        self.entity_extractor = EntityExtractor(config)
        self.query_processor = QueryProcessor(config)
        self._executor = ThreadPoolExecutor(max_workers=config.max_workers)

    def search(self, repo_id: str, query: str,
               strategy: SearchStrategy = SearchStrategy.HYBRID,
               graph: Optional[nx.DiGraph] = None) -> Dict[str, Any]:
        """Perform advanced search with specified strategy."""
        start_time = time.time()

        # Check cache
        cache_key = self._generate_cache_key(repo_id, query, strategy)
        cached_result = search_cache.get(cache_key)
        if cached_result:
            logger.info(f"Cache hit for query: {query}")
            return cached_result

        # Extract entities
        entities = self.entity_extractor.extract(query)

        # Choose search strategy
        if strategy == SearchStrategy.SIMPLE:
            results = self._simple_search(repo_id, query)
        elif strategy == SearchStrategy.EXPANDED:
            results = self._expanded_search(repo_id, query, entities)
        elif strategy == SearchStrategy.MULTI_HOP:
            results = self._multi_hop_search(repo_id, query, entities)
        elif strategy == SearchStrategy.GRAPH_GUIDED and graph:
            results = self._graph_guided_search(repo_id, query, entities, graph)
        else:
            results = self._hybrid_search(repo_id, query, entities, graph)

        # Post-process results
        results = self._post_process_results(results, query, entities)

        # Add metadata
        results['metadata'] = {
            'strategy': strategy.value,
            'entities_found': {k: len(v) for k, v in entities.items()},
            'search_time': time.time() - start_time,
            'total_results': len(results.get('chunks', []))
        }

        # Cache results
        search_cache.put(cache_key, results)

        return results

    def _simple_search(self, repo_id: str, query: str) -> Dict[str, Any]:
        """Perform simple single-query search."""
        try:
            chunks = ollama.search_index(
                query,
                "snowflake-arctic-embed2:latest",
                repo_id,
                k=self.config.default_k
            )
            return {'chunks': chunks or []}
        except Exception as e:
            logger.error(f"Simple search failed: {e}")
            return {'chunks': [], 'error': str(e)}

    def _expanded_search(self, repo_id: str, query: str,
                        entities: Dict[str, List[Entity]]) -> Dict[str, Any]:
        """Perform expanded search with query variations."""
        expansions = self.query_processor.process(query, entities, SearchStrategy.EXPANDED)

        all_chunks = []
        futures = []

        for expansion in expansions:
            future = self._executor.submit(
                ollama.search_index,
                expansion,
                "snowflake-arctic-embed2:latest",
                repo_id,
                k=self.config.default_k // len(expansions)
            )
            futures.append((future, expansion))

        for future, expansion in futures:
            try:
                chunks = future.result(timeout=self.config.timeout_seconds)
                if chunks:
                    for chunk in chunks:
                        chunk['query_source'] = expansion
                    all_chunks.extend(chunks)
            except Exception as e:
                logger.warning(f"Search failed for expansion '{expansion}': {e}")

        return {'chunks': all_chunks, 'expansions': expansions}

    def _multi_hop_search(self, repo_id: str, query: str,
                         entities: Dict[str, List[Entity]]) -> Dict[str, Any]:
        """Perform multi-hop search following references."""
        all_chunks = []
        hop_results = []
        visited_queries = set()

        current_queries = [query]
        current_entities = entities

        for hop in range(self.config.max_hops):
            if not current_queries:
                break

            hop_chunks = []
            next_queries = []
            next_entities = defaultdict(list)

            # Search current queries
            for q in current_queries:
                if q in visited_queries:
                    continue

                visited_queries.add(q)

                # Search
                result = self._expanded_search(repo_id, q, current_entities)
                chunks = result.get('chunks', [])

                if chunks:
                    # Apply hop decay to scores
                    decay = self.config.hop_decay ** hop
                    for chunk in chunks:
                        chunk['hop'] = hop
                        chunk['relevance_score'] = chunk.get('relevance_score', 1.0) * decay

                    hop_chunks.extend(chunks)

                    # Extract new entities for next hop
                    for chunk in chunks[:5]:  # Analyze top chunks
                        text = chunk.get('text', '')
                        new_entities = self.entity_extractor.extract(text, use_llm=False)
                        for entity_type, entity_list in new_entities.items():
                            next_entities[entity_type].extend(entity_list)

            if hop_chunks:
                all_chunks.extend(hop_chunks)
                hop_results.append({
                    'hop': hop,
                    'queries': current_queries,
                    'chunks_found': len(hop_chunks)
                })

            # Generate next hop queries
            if hop < self.config.max_hops - 1:
                for entity_type, entity_list in next_entities.items():
                    for entity in entity_list[:2]:  # Limit expansion
                        if entity.confidence > self.config.min_hop_relevance:
                            next_queries.append(entity.text)

            current_queries = list(set(next_queries))[:self.config.max_expansions]
            current_entities = dict(next_entities)

        return {
            'chunks': all_chunks,
            'hop_results': hop_results,
            'total_hops': len(hop_results)
        }

    def _graph_guided_search(self, repo_id: str, query: str,
                           entities: Dict[str, List[Entity]],
                           graph: nx.DiGraph) -> Dict[str, Any]:
        """Use graph structure to guide search."""
        # Find relevant nodes in graph
        relevant_nodes = self._find_relevant_nodes(graph, entities)

        if not relevant_nodes:
            # Fallback to expanded search
            return self._expanded_search(repo_id, query, entities)

        all_chunks = []
        node_results = []

        # Search for each relevant node and its neighbors
        for node_id, relevance_score in relevant_nodes[:10]:
            node_data = graph.nodes[node_id]

            # Generate targeted query
            node_query = self._generate_node_query(node_id, node_data, query)

            # Search
            chunks = ollama.search_index(
                node_query,
                "snowflake-arctic-embed2:latest",
                repo_id,
                k=5
            )

            if chunks:
                for chunk in chunks:
                    chunk['graph_node'] = node_id
                    chunk['node_relevance'] = relevance_score

                all_chunks.extend(chunks)

                node_results.append({
                    'node': node_id,
                    'query': node_query,
                    'chunks_found': len(chunks)
                })

        return {
            'chunks': all_chunks,
            'node_results': node_results,
            'graph_nodes_searched': len(relevant_nodes)
        }

    def _hybrid_search(self, repo_id: str, query: str,
                      entities: Dict[str, List[Entity]],
                      graph: Optional[nx.DiGraph] = None) -> Dict[str, Any]:
        """Combine multiple search strategies."""
        all_results = []

        # Parallel execution of different strategies
        futures = [
            (self._executor.submit(self._expanded_search, repo_id, query, entities), 'expanded'),
            (self._executor.submit(self._multi_hop_search, repo_id, query, entities), 'multi_hop')
        ]

        if graph:
            futures.append(
                (self._executor.submit(self._graph_guided_search, repo_id, query, entities, graph), 'graph_guided')
            )

        strategy_results = {}
        all_chunks = []

        for future, strategy_name in futures:
            try:
                result = future.result(timeout=self.config.timeout_seconds)
                strategy_results[strategy_name] = result
                all_chunks.extend(result.get('chunks', []))
            except Exception as e:
                logger.warning(f"Strategy {strategy_name} failed: {e}")

        # Merge and deduplicate results
        unique_chunks = self._deduplicate_chunks(all_chunks)

        return {
            'chunks': unique_chunks,
            'strategy_results': strategy_results,
            'strategies_used': list(strategy_results.keys())
        }

    def _find_relevant_nodes(self, graph: nx.DiGraph,
                           entities: Dict[str, List[Entity]]) -> List[Tuple[str, float]]:
        """Find nodes in graph relevant to entities."""
        node_scores = []

        for node_id, node_data in graph.nodes(data=True):
            score = 0.0

            # Score based on entity matches
            for entity_type, entity_list in entities.items():
                for entity in entity_list:
                    if entity.text.lower() in node_id.lower():
                        score += entity.confidence * 2
                    if entity.text.lower() in str(node_data.get('name', '')).lower():
                        score += entity.confidence

                    # Check aliases
                    for alias in entity.aliases:
                        if alias.lower() in node_id.lower():
                            score += entity.confidence * 0.5

            if score > 0:
                node_scores.append((node_id, score))

        # Sort by score
        node_scores.sort(key=lambda x: x[1], reverse=True)
        return node_scores

    def _generate_node_query(self, node_id: str, node_data: Dict[str, Any],
                           original_query: str) -> str:
        """Generate a query specific to a graph node."""
        node_name = node_data.get('name', node_id)
        node_type = node_data.get('type', '')
        file_path = node_data.get('file', '')

        # Combine node info with original query intent
        if file_path:
            return f"{node_name} in {file_path} {original_query}"
        else:
            return f"{node_type} {node_name} {original_query}"

    def _post_process_results(self, results: Dict[str, Any], query: str,
                            entities: Dict[str, List[Entity]]) -> Dict[str, Any]:
        """Post-process search results with reranking and clustering."""
        chunks = results.get('chunks', [])

        if not chunks:
            return results

        # Deduplicate
        chunks = self._deduplicate_chunks(chunks)

        # Rerank
        chunks = self._rerank_chunks(chunks, query, entities)

        # Cluster if enabled
        if self.config.use_semantic_clustering and len(chunks) > 10:
            clusters = self._cluster_chunks(chunks)
            results['clusters'] = clusters

        # Select top chunks with diversity
        final_chunks = self._select_diverse_chunks(chunks, self.config.default_k)

        results['chunks'] = final_chunks
        results['total_before_filtering'] = len(chunks)

        return results

    def _deduplicate_chunks(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Advanced deduplication using content similarity."""
        if not chunks:
            return []

        unique_chunks = []
        seen_hashes = set()
        seen_content = []

        for chunk in chunks:
            # Quick hash check
            chunk_hash = self._calculate_chunk_hash(chunk)
            if chunk_hash in seen_hashes:
                continue

            # Semantic similarity check
            if self._is_duplicate(chunk, seen_content):
                continue

            seen_hashes.add(chunk_hash)
            seen_content.append(chunk)
            unique_chunks.append(chunk)

        return unique_chunks

    def _calculate_chunk_hash(self, chunk: Dict[str, Any]) -> str:
        """Calculate hash for chunk."""
        text = chunk.get('text', '')
        file_path = chunk.get('file_path', '')
        # Use more of the content for better deduplication
        content = f"{file_path}:{text[:200]}"
        return hashlib.md5(content.encode()).hexdigest()

    def _is_duplicate(self, chunk: Dict[str, Any],
                     seen_content: List[Dict[str, Any]]) -> bool:
        """Check if chunk is semantically similar to seen content."""
        chunk_text = chunk.get('text', '').lower()
        chunk_words = set(chunk_text.split())

        for seen_chunk in seen_content:
            seen_text = seen_chunk.get('text', '').lower()
            seen_words = set(seen_text.split())

            # Calculate Jaccard similarity
            intersection = len(chunk_words & seen_words)
            union = len(chunk_words | seen_words)

            if union > 0:
                similarity = intersection / union
                if similarity > self.config.similarity_threshold:
                    return True

        return False

    def _rerank_chunks(self, chunks: List[Dict[str, Any]], query: str,
                      entities: Dict[str, List[Entity]]) -> List[Dict[str, Any]]:
        """Advanced reranking with multiple signals."""
        # Extract ranking features
        for chunk in chunks:
            features = self._extract_ranking_features(chunk, query, entities)

            # Calculate composite score
            score = (
                features['text_relevance'] * 0.3 +
                features['entity_coverage'] * 0.3 +
                features['code_quality'] * 0.2 +
                features['freshness'] * 0.1 +
                features['structural_relevance'] * 0.1
            )

            # Apply existing scores
            if 'relevance_score' in chunk:
                score = score * 0.7 + chunk['relevance_score'] * 0.3

            chunk['final_score'] = score
            chunk['ranking_features'] = features

        # Sort by final score
        chunks.sort(key=lambda x: x.get('final_score', 0), reverse=True)

        return chunks

    def _extract_ranking_features(self, chunk: Dict[str, Any], query: str,
                                entities: Dict[str, List[Entity]]) -> Dict[str, float]:
        """Extract features for ranking."""
        text = chunk.get('text', '').lower()
        file_path = chunk.get('file_path', '').lower()

        features = {
            'text_relevance': self._calculate_text_relevance(text, query),
            'entity_coverage': self._calculate_entity_coverage(text, entities),
            'code_quality': self._estimate_code_quality(text),
            'freshness': self._calculate_freshness(chunk),
            'structural_relevance': self._calculate_structural_relevance(chunk)
        }

        return features

    def _calculate_text_relevance(self, text: str, query: str) -> float:
        """Calculate text relevance to query."""
        query_terms = set(query.lower().split())
        text_terms = set(text.split())

        # Term frequency
        term_freq = sum(1 for term in query_terms if term in text_terms)

        # Normalize by query length
        relevance = term_freq / len(query_terms) if query_terms else 0

        # Boost for exact phrase match
        if query.lower() in text:
            relevance += 0.5

        return min(relevance, 1.0)

    def _calculate_entity_coverage(self, text: str,
                                 entities: Dict[str, List[Entity]]) -> float:
        """Calculate how many entities are covered in text."""
        total_entities = sum(len(v) for v in entities.values())
        if total_entities == 0:
            return 0.5  # Neutral score

        covered = 0
        for entity_list in entities.values():
            for entity in entity_list:
                if entity.text.lower() in text:
                    covered += entity.confidence
                # Check aliases
                for alias in entity.aliases:
                    if alias.lower() in text:
                        covered += entity.confidence * 0.5
                        break

        return min(covered / total_entities, 1.0)

    def _estimate_code_quality(self, text: str) -> float:
        """Estimate code quality based on heuristics."""
        quality_score = 0.5  # Base score

        # Check for code indicators
        code_indicators = [
            (r'def\s+\w+\s*\(', 0.1),  # Functions
            (r'class\s+\w+', 0.1),      # Classes
            (r'import\s+\w+', 0.05),    # Imports
            (r'"""[\s\S]+?"""', 0.1),   # Docstrings
            (r'#\s*\w+', 0.05),         # Comments
        ]

        for pattern, score in code_indicators:
            if re.search(pattern, text):
                quality_score += score

        # Penalty for too long or too short
        length = len(text)
        if length < 50:
            quality_score *= 0.5
        elif length > 2000:
            quality_score *= 0.8

        return min(quality_score, 1.0)

    def _calculate_freshness(self, chunk: Dict[str, Any]) -> float:
        """Calculate freshness score (placeholder for timestamp-based scoring)."""
        # Could be enhanced with actual file timestamps
        return 0.5

    def _calculate_structural_relevance(self, chunk: Dict[str, Any]) -> float:
        """Calculate relevance based on structural properties."""
        score = 0.5

        # Boost for certain file types
        file_path = chunk.get('file_path', '').lower()
        if file_path.endswith(('.py', '.js', '.java')):
            score += 0.2
        elif file_path.endswith(('.md', '.txt', '.rst')):
            score += 0.1

        # Boost for specific directories
        important_dirs = ['src', 'core', 'api', 'routes', 'models', 'services']
        for dir_name in important_dirs:
            if dir_name in file_path:
                score += 0.1
                break

        return min(score, 1.0)

    def _cluster_chunks(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Cluster chunks by semantic similarity."""
        # Simple clustering based on shared entities
        clusters = []
        clustered = set()

        for i, chunk in enumerate(chunks):
            if i in clustered:
                continue

            cluster = {
                'id': f'cluster_{i}',
                'chunks': [chunk],
                'representative': chunk,
                'size': 1
            }

            # Find similar chunks
            for j, other_chunk in enumerate(chunks[i+1:], i+1):
                if j in clustered:
                    continue

                if self._chunks_similar(chunk, other_chunk):
                    cluster['chunks'].append(other_chunk)
                    cluster['size'] += 1
                    clustered.add(j)

            clusters.append(cluster)

        return clusters

    def _chunks_similar(self, chunk1: Dict[str, Any],
                       chunk2: Dict[str, Any]) -> bool:
        """Check if two chunks are similar enough to cluster."""
        # File-based clustering
        if chunk1.get('file_path') == chunk2.get('file_path'):
            return True

        # Content-based clustering
        text1 = set(chunk1.get('text', '').lower().split())
        text2 = set(chunk2.get('text', '').lower().split())

        if len(text1) == 0 or len(text2) == 0:
            return False

        overlap = len(text1 & text2)
        smaller = min(len(text1), len(text2))

        return (overlap / smaller) > 0.5

    def _select_diverse_chunks(self, chunks: List[Dict[str, Any]], k: int) -> List[Dict[str, Any]]:
        """Select top k chunks with diversity."""
        if len(chunks) <= k:
            return chunks

        selected = []
        remaining = chunks.copy()

        # Select best chunk
        if remaining:
            selected.append(remaining.pop(0))

        # Select diverse chunks
        while len(selected) < k and remaining:
            # Find chunk most different from selected
            best_chunk = None
            best_diversity = -1

            for chunk in remaining:
                diversity = self._calculate_diversity(chunk, selected)
                if diversity > best_diversity:
                    best_diversity = diversity
                    best_chunk = chunk

            if best_chunk:
                selected.append(best_chunk)
                remaining.remove(best_chunk)

        return selected

    def _calculate_diversity(self, chunk: Dict[str, Any],
                           selected: List[Dict[str, Any]]) -> float:
        """Calculate diversity of chunk compared to selected chunks."""
        if not selected:
            return 1.0

        chunk_text = set(chunk.get('text', '').lower().split())

        min_similarity = 1.0
        for sel_chunk in selected:
            sel_text = set(sel_chunk.get('text', '').lower().split())

            if len(chunk_text) == 0 or len(sel_text) == 0:
                similarity = 0
            else:
                overlap = len(chunk_text & sel_text)
                union = len(chunk_text | sel_text)
                similarity = overlap / union if union > 0 else 0

            min_similarity = min(min_similarity, similarity)

        return 1.0 - min_similarity

    def _generate_cache_key(self, repo_id: str, query: str,
                          strategy: SearchStrategy) -> str:
        """Generate cache key for search results."""
        content = f"{repo_id}:{query}:{strategy.value}"
        return hashlib.md5(content.encode()).hexdigest()

    def close(self):
        """Clean up resources."""
        self._executor.shutdown(wait=True)

# --- Graph Analysis ---
def _build_knowledge_graph(graph_data: Dict[str, Any]) -> nx.DiGraph:
    """Builds a networkx DiGraph from the architectural graph data."""
    if not isinstance(graph_data, dict):
        raise GraphBuildingError(f"Graph data must be a dictionary, got {type(graph_data)}")

    try:
        G = nx.DiGraph()
        nodes = graph_data.get("nodes", {})
        edges = graph_data.get("edges", [])

        if not isinstance(nodes, dict):
            logger.warning(f"Nodes data is not a dictionary, got {type(nodes)}. Using empty dict.")
            nodes = {}

        if not isinstance(edges, list):
            logger.warning(f"Edges data is not a list, got {type(edges)}. Using empty list.")
            edges = []

        # Add nodes with validation
        nodes_added = 0
        for node_id, node_data in nodes.items():
            try:
                if not isinstance(node_id, str):
                    logger.warning(f"Skipping node with invalid ID type: {type(node_id)}")
                    continue

                if not isinstance(node_data, dict):
                    logger.warning(f"Node {node_id} has invalid data type: {type(node_data)}. Using empty dict.")
                    node_data = {}

                G.add_node(node_id, **node_data)
                nodes_added += 1
            except Exception as e:
                logger.warning(f"Failed to add node {node_id}: {e}")
                continue

        # Add edges with validation
        edges_added = 0
        for edge in edges:
            try:
                if not isinstance(edge, dict):
                    logger.warning(f"Skipping invalid edge: {edge}")
                    continue

                source = edge.get("source")
                target = edge.get("target")

                if not source or not target:
                    logger.warning(f"Edge missing source or target: {edge}")
                    continue

                if not isinstance(source, str) or not isinstance(target, str):
                    logger.warning(f"Edge source/target must be strings: {edge}")
                    continue

                # Only add edge if source exists in graph
                if source in G:
                    edge_type = edge.get("type", "unknown")
                    G.add_edge(source, target, type=edge_type)
                    edges_added += 1
                else:
                    logger.warning(f"Source node {source} not found in graph for edge to {target}")
            except Exception as e:
                logger.warning(f"Failed to add edge {edge}: {e}")
                continue

        logger.info(f"Built knowledge graph with {nodes_added} nodes and {edges_added} edges.")

        if nodes_added == 0:
            raise GraphBuildingError("No valid nodes were added to the graph")

        return G

    except Exception as e:
        if isinstance(e, GraphBuildingError):
            raise
        raise GraphBuildingError(f"Failed to build knowledge graph: {e}")

# --- Context Building ---
class ContextBuilder:
    """Builds context for LLM synthesis from search results."""

    def __init__(self, config: RAGConfig):
        self.config = config

    def build(self, search_results: Dict[str, Any],
              question: str,
              is_specific: bool = False) -> str:
        """Build context string from search results."""
        chunks = search_results.get('chunks', [])

        if not chunks:
            return "No relevant code context was found for your question."

        # Group chunks by file for better organization
        chunks_by_file = defaultdict(list)
        for chunk in chunks:
            file_path = chunk.get('file_path', 'unknown')
            chunks_by_file[file_path].append(chunk)

        # Build context with structure
        context_parts = []

        # Add metadata if available
        if 'metadata' in search_results:
            meta = search_results['metadata']
            context_parts.append(f"[Search performed using {meta.get('strategy', 'unknown')} strategy, found {meta.get('total_results', 0)} results in {meta.get('search_time', 0):.2f}s]")

        # Add clustered results if available
        if 'clusters' in search_results:
            context_parts.append(self._format_clusters(search_results['clusters']))
        else:
            # Format chunks by file
            for file_path, file_chunks in chunks_by_file.items():
                context_parts.append(self._format_file_chunks(file_path, file_chunks))

        # Add hop information if multi-hop search
        if 'hop_results' in search_results:
            context_parts.append(self._format_hop_results(search_results['hop_results']))

        return "\n\n".join(context_parts)

    def _format_file_chunks(self, file_path: str, chunks: List[Dict[str, Any]]) -> str:
        """Format chunks from a single file."""
        parts = [f"=== File: {file_path} ==="]

        for i, chunk in enumerate(chunks[:5]):  # Limit chunks per file
            text = chunk.get('text', '').strip()
            if text:
                parts.append(f"\n[Chunk {i+1}]")
                if 'final_score' in chunk:
                    parts.append(f"Relevance: {chunk['final_score']:.2f}")
                parts.append(text)

        return "\n".join(parts)

    def _format_clusters(self, clusters: List[Dict[str, Any]]) -> str:
        """Format clustered results."""
        parts = ["=== Clustered Results ==="]

        for cluster in clusters[:5]:  # Limit clusters
            parts.append(f"\n[Cluster: {cluster['id']} - {cluster['size']} related chunks]")

            # Show representative chunk
            rep = cluster['representative']
            parts.append(f"File: {rep.get('file_path', 'unknown')}")
            parts.append(rep.get('text', '')[:500])  # Truncate

            if cluster['size'] > 1:
                parts.append(f"... and {cluster['size'] - 1} similar chunks")

        return "\n".join(parts)

    def _format_hop_results(self, hop_results: List[Dict[str, Any]]) -> str:
        """Format multi-hop search results."""
        parts = ["=== Multi-hop Search Path ==="]

        for hop in hop_results:
            parts.append(f"\nHop {hop['hop'] + 1}:")
            parts.append(f"Queries: {', '.join(hop['queries'][:3])}")
            parts.append(f"Found: {hop['chunks_found']} results")

        return "\n".join(parts)

# --- Main Orchestration ---
class OracleService:
    """Enhanced Oracle service with PhD-level RAG capabilities."""

    def __init__(self, config: Optional[RAGConfig] = None):
        self.config = config or rag_config
        self.search_orchestrator = SearchOrchestrator(self.config)
        self.context_builder = ContextBuilder(self.config)
        self.entity_extractor = EntityExtractor(self.config)
    
    def _load_data(self, repo_or_archive_id: str, is_archive: bool) -> Dict[str, Any]:
        """
        Load cortex data for either a repository or archive.
        
        Args:
            repo_or_archive_id: ID of the repo or archive
            is_archive: Whether this is an archive or repository
            
        Returns:
            Cortex data dictionary
            
        Raises:
            Exception: If loading fails
        """
        if is_archive:
            # Load from local_archives directory
            from pathlib import Path
            backend_dir = Path(__file__).resolve().parent.parent.parent
            archives_dir = backend_dir / "local_archives"
            cortex_path = archives_dir / repo_or_archive_id / f"{repo_or_archive_id}_cortex.json"
            
            if not cortex_path.exists():
                raise cortex_service.CortexFileNotFound(f"Archive cortex file not found: {cortex_path}")
            
            try:
                import json
                with open(cortex_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except json.JSONDecodeError:
                raise cortex_service.CortexFileMalformed(f"Archive cortex file is corrupted: {cortex_path}")
        else:
            # Use existing cortex service for repositories
            return cortex_service.load_cortex_data(repo_or_archive_id)
    
    def _answer_question_simple_rag(self, archive_id: str, question: str, cortex_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Answer questions for archives using simplified RAG (no architectural graph).
        
        Args:
            archive_id: Archive identifier
            question: User question
            cortex_data: Cortex data for the archive
            
        Returns:
            Answer response dictionary
        """
        try:
            # Use basic RAG search without graph analysis
            chunks = ollama.search_index(
                query=question,
                repo_id=archive_id,
                artifact_type='archive'  # Tell ollama to look in local_archives
            )
            
            if not chunks:
                return {
                    "answer": "I couldn't find any relevant information in this archive to answer your question.",
                    "confidence_score": 0.1,
                    "source_chunks": [],
                    "archive_id": archive_id,
                    "search_strategy": "simple_rag"
                }
            
            # Build context from chunks
            context_parts = []
            source_info = []
            
            for chunk in chunks[:5]:  # Use top 5 chunks
                content = chunk.get('content', '')
                file_path = chunk.get('file_path', 'unknown')
                score = chunk.get('score', 0.0)
                
                if content.strip():
                    context_parts.append(f"From {file_path}:\n{content}")
                    source_info.append({
                        "file_path": file_path,
                        "similarity_score": score,
                        "chunk_preview": content[:200] + "..." if len(content) > 200 else content
                    })
            
            context = "\n\n".join(context_parts)
            
            # Generate answer using LLM
            prompt = f"""Based on the following information from a document archive, please answer the user's question.

Archive Content:
{context}

User Question: {question}

Please provide a helpful and accurate answer based on the provided content. If the content doesn't contain enough information to answer the question fully, please say so and suggest what additional information might be needed."""

            answer = llm_service.generate_text(prompt, task_type=TaskType.ANALYSIS)
            
            # Calculate confidence based on chunk scores
            avg_score = sum(chunk.get('score', 0.0) for chunk in chunks[:5]) / len(chunks[:5]) if chunks else 0.0
            confidence = min(0.9, max(0.3, avg_score))
            
            return {
                "answer": answer,
                "confidence_score": confidence,
                "source_chunks": source_info,
                "archive_id": archive_id,
                "search_strategy": "simple_rag",
                "archive_metadata": cortex_data.get("archive_metadata", {})
            }
            
        except Exception as e:
            logger.error(f"Error in simple RAG for archive {archive_id}: {e}")
            return {
                "error": f"Failed to process question for archive: {str(e)}",
                "archive_id": archive_id
            }

    def answer_question(self, repo_or_archive_id: str, question: str,
                       use_enhanced_rag: bool = True) -> Dict[str, Any]:
        """
        Main entry point for answering questions with enhanced RAG.
        Supports both repositories and archives (Librarian's Archives feature).
        Maintains backward compatibility while adding advanced features.
        """
        # Input validation
        if not repo_or_archive_id or not isinstance(repo_or_archive_id, str):
            return {"error": "Invalid repository/archive ID provided"}

        if not question or not isinstance(question, str):
            return {"error": "Invalid question provided"}

        # Determine if this is a repository or archive
        is_archive = repo_or_archive_id.startswith('archive_')
        data_type = "archive" if is_archive else "repository"
        
        logger.info(f"The Oracle received a question for {data_type} '{repo_or_archive_id}': '{question}'")

        # Load repository/archive data
        try:
            cortex_data = self._load_data(repo_or_archive_id, is_archive)
            if not isinstance(cortex_data, dict):
                return {"error": "Invalid cortex data format"}

            graph_data = cortex_data.get("architectural_graph")
            if not graph_data:
                if is_archive:
                    # Archives may not have architectural graphs - use simplified RAG
                    return self._answer_question_simple_rag(repo_or_archive_id, question, cortex_data)
                else:
                    return {"error": "Architectural graph not found for this repository."}

        except cortex_service.CortexFileNotFound:
            return {"error": f"Cortex file for {data_type} '{repo_or_archive_id}' not found. Please ingest it first."}
        except cortex_service.CortexFileMalformed:
            return {"error": f"Cortex file for {data_type} '{repo_or_archive_id}' is corrupted. Please re-ingest it."}
        except Exception as e:
            logger.error(f"Error loading cortex data for {data_type} {repo_or_archive_id}: {e}")
            return {"error": f"Failed to load {data_type} data. Please try again later."}

        # Build graph
        try:
            graph = _build_knowledge_graph(graph_data)
        except GraphBuildingError as e:
            logger.error(f"Graph building failed: {e}")
            return {"error": "Failed to analyze repository structure. The repository data may be corrupted."}
        except Exception as e:
            logger.error(f"Unexpected error building graph: {e}")
            return {"error": "An unexpected error occurred while analyzing the repository structure."}

        # Extract entities
        try:
            entities_dict = self.entity_extractor.extract(question)
            # Convert to old format for compatibility
            entities = self._convert_entities_format(entities_dict)
        except Exception as e:
            logger.error(f"Entity extraction failed: {e}")
            entities = self._get_default_entities()

        # Determine search strategy
        strategy = self._determine_strategy(question, entities, use_enhanced_rag)

        # Perform search
        try:
            search_results = self.search_orchestrator.search(
                repo_id, question, strategy, graph
            )
        except Exception as e:
            logger.error(f"Search failed: {e}")
            search_results = {'chunks': [], 'error': str(e)}

        # Build context
        context_string = self.context_builder.build(
            search_results, question,
            is_specific=self._is_specific_question(entities)
        )

        # Determine prompt heading
        if self._is_specific_question(entities):
            prompt_heading = "ARCHITECTURAL IMPACT ANALYSIS"
        else:
            prompt_heading = "RELEVANT CODE & COMMENTARY"

        # Generate final answer
        try:
            answer = self._synthesize_answer(
                question, context_string, prompt_heading,
                entities, search_results
            )

            response = {"answer": answer}

            # Add metadata if enhanced RAG was used
            if use_enhanced_rag and 'metadata' in search_results:
                response["search_metadata"] = search_results['metadata']

            return response

        except Exception as e:
            logger.error(f"Final synthesis failed: {e}")
            return {"error": "An unexpected error occurred while generating the final answer."}

    def _determine_strategy(self, question: str, entities: Dict[str, List[str]],
                          use_enhanced: bool) -> SearchStrategy:
        """Determine the best search strategy for the question."""
        if not use_enhanced:
            return SearchStrategy.SIMPLE

        # Check question complexity
        question_lower = question.lower()

        # Multi-hop indicators
        if any(word in question_lower for word in ["how does", "workflow", "process", "trace"]):
            return SearchStrategy.MULTI_HOP

        # Graph-guided indicators
        if any(word in question_lower for word in ["impact", "depends", "affects", "uses"]):
            return SearchStrategy.GRAPH_GUIDED

        # Complex questions need hybrid approach
        if len(question.split()) > 15 or len(entities.get('general_terms', [])) > 3:
            return SearchStrategy.HYBRID

        # Default to expanded search
        return SearchStrategy.EXPANDED

    def _convert_entities_format(self, entities_dict: Dict[str, List[Entity]]) -> Dict[str, List[str]]:
        """Convert new entity format to old format for compatibility."""
        old_format = {}
        for entity_type, entity_list in entities_dict.items():
            old_format[entity_type] = [e.text for e in entity_list]
        return old_format

    def _get_default_entities(self) -> Dict[str, List[str]]:
        """Returns default empty entity structure."""
        return {
            "functions": [],
            "classes": [],
            "files": [],
            "general_terms": [],
            "code_patterns": [],
            "technologies": []
        }

    def _is_specific_question(self, entities: Dict[str, List[str]]) -> bool:
        """Check if question is about specific code entities."""
        return any(entities.get(k) for k in ["functions", "classes", "files"])

    def _synthesize_answer(self, question: str, context: str,
                         prompt_heading: str, entities: Dict[str, List[str]],
                         search_results: Dict[str, Any]) -> str:
        """Synthesize final answer using LLM."""
        # Self-reflection prompt if enabled
        reflection = ""
        if self.config.enable_self_reflection and search_results.get('error'):
            reflection = f"""
**Search Challenges:**
The search encountered some difficulties: {search_results.get('error')}
This may affect the completeness of the answer.
"""

        synthesis_prompt = f"""You are The Oracle, an advanced AI system that understands code architecture at a PhD level.
Your purpose is to provide deep, insightful answers by combining architectural graph analysis with comprehensive source code understanding.

**DEVELOPER'S QUESTION:**
{question}

**{prompt_heading}:**
{context}

**EXTRACTED ENTITIES:**
- Functions: {entities.get('functions', [])}
- Classes: {entities.get('classes', [])}
- Files: {entities.get('files', [])}
- Technologies: {entities.get('technologies', [])}
- Patterns: {entities.get('code_patterns', [])}

{reflection}

---
Based on ALL the evidence above, provide a comprehensive answer that:
1. **Directly answers** the developer's question with specific details
2. **Cites evidence** from the code snippets to support your answer
3. **Explains architectural implications** and design patterns observed
4. **Identifies potential issues** or areas for improvement
5. **Suggests next steps** for deeper investigation if needed

Use clear markdown formatting with headers, code blocks, and lists where appropriate.
Be specific and technical, but also explain the "why" behind the code structure.
"""

        answer = llm_service.generate_text(synthesis_prompt, task_type=TaskType.COMPLEX_REASONING)
        return answer

    def close(self):
        """Clean up resources."""
        self.search_orchestrator.close()

# --- Global Oracle Instance ---
_oracle_instance = None

def get_oracle() -> OracleService:
    """Get or create the global Oracle instance."""
    global _oracle_instance
    if _oracle_instance is None:
        _oracle_instance = OracleService()
    return _oracle_instance

# --- Public API (Backward Compatible) ---
def answer_question(repo_id: str, question: str, use_enhanced_rag: bool = True) -> Dict[str, Any]:
    """
    Public API for answering questions. Maintains backward compatibility.
    """
    oracle = get_oracle()
    return oracle.answer_question(repo_id, question, use_enhanced_rag)

def answer_question_legacy(repo_id: str, question: str) -> Dict[str, Any]:
    """
    Legacy wrapper that uses the original RAG behavior.
    For backward compatibility with existing code.
    """
    return answer_question(repo_id, question, use_enhanced_rag=False)

def perform_semantic_search(repo_id: str, query: str, search_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Public interface for performing semantic search with enhanced RAG capabilities.
    """
    oracle = get_oracle()
    config = search_config or {}

    # Determine strategy from config
    strategy = SearchStrategy.EXPANDED
    if config.get('use_multi_hop'):
        strategy = SearchStrategy.MULTI_HOP
    elif config.get('use_graph'):
        strategy = SearchStrategy.GRAPH_GUIDED

    # Perform search
    results = oracle.search_orchestrator.search(repo_id, query, strategy)

    # Format for API
    return {
        "results": results.get('chunks', [])[:config.get('k', 15)],
        "total": len(results.get('chunks', [])),
        "query": query,
        "metadata": results.get('metadata', {})
    }

def analyze_code_impact(repo_id: str, file_path: str, function_or_class: str) -> Dict[str, Any]:
    """
    Analyzes the impact of changes to a specific code entity.
    """
    question = f"What is the impact of changing {function_or_class} in {file_path}?"
    oracle = get_oracle()

    # Use graph-guided search for impact analysis
    result = oracle.answer_question(repo_id, question, use_enhanced_rag=True)

    # Extract specific impact information
    return {
        "entity": function_or_class,
        "file": file_path,
        "analysis": result.get("answer", ""),
        "metadata": result.get("search_metadata", {})
    }

# --- Configuration Management ---
def update_config(new_config: Dict[str, Any]) -> None:
    """Update RAG configuration."""
    global rag_config
    for key, value in new_config.items():
        if hasattr(rag_config, key):
            setattr(rag_config, key, value)

    # Clear cache on config change
    search_cache.clear()

def get_config() -> RAGConfig:
    """Get current RAG configuration."""
    return rag_config

# --- Cleanup ---
def cleanup():
    """Clean up resources on shutdown."""
    global _oracle_instance
    if _oracle_instance:
        _oracle_instance.close()
        _oracle_instance = None

# Register cleanup
import atexit
atexit.register(cleanup)
