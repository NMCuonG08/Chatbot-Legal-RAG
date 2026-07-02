"""
Enhanced Search Module for Vietnamese Legal Chatbot
Implements hybrid search combining semantic vector search + BM25 keyword search using LlamaIndex
"""

import logging
import pickle
import re
from pathlib import Path
from typing import Dict, List, Optional

from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.retrievers.bm25 import BM25Retriever

from brain import get_embedding
from config import DEFAULT_COLLECTION_NAME
from vectorize import search_vector

logger = logging.getLogger(__name__)

# Global search components
_docstore = None
_bm25_retriever = None
_search_engine_initialized = False


def initialize_search_index(documents: List[Dict]) -> bool:
    """
    Initialize BM25 search index from documents
    
    Args:
        documents: List of documents with keys: question, content, source, doc_id
        
    Returns:
        bool: True if successful, False otherwise
    """
    global _docstore, _bm25_retriever, _search_engine_initialized
    
    try:
        logger.info(f"🔧 Initializing search index with {len(documents)} documents")
        
        # Convert documents directly to TextNode objects to bypass slow SentenceSplitter parsing
        from llama_index.core.schema import TextNode
        nodes = []
        for i, doc in enumerate(documents):
            if not doc.get('question') and not doc.get('content'):
                continue
            text = f"{doc.get('question', '')} {doc.get('content', '')}"
            node = TextNode(
                text=text,
                id_=str(doc.get('doc_id', i)),
                metadata={
                    "question": doc.get('question', ''),
                    "source": doc.get('source', 'unknown'),
                    "doc_id": doc.get('doc_id', i)
                }
            )
            nodes.append(node)
        
        if not nodes:
            logger.error("❌ No valid documents after conversion")
            return False
            
        logger.info(f"📄 Converted {len(nodes)} valid documents directly to TextNodes")
        
        # Initialize docstore
        _docstore = SimpleDocumentStore()
        _docstore.add_documents(nodes)
        logger.info(f"📚 Docstore initialized with {len(nodes)} nodes")
        
        # Initialize BM25 retriever without stemmer for simplicity
        _bm25_retriever = BM25Retriever.from_defaults(
            docstore=_docstore,
            similarity_top_k=5,
        )
        logger.info("🔍 BM25 retriever initialized successfully")
        
        _search_engine_initialized = True
        logger.info("✅ Search index initialized successfully!")
        
        # Verify initialization
        stats = get_search_stats()
        logger.info(f"📊 Search stats: {stats}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize search index: {e}")
        logger.exception("Full error traceback:")
        _search_engine_initialized = False
        return False


def hybrid_search(query: str, limit: int = 10) -> List[Dict]:
    """
    Perform hybrid search combining BM25 keyword search and vector semantic search
    
    Args:
        query: Search query
        limit: Maximum number of results to return
        
    Returns:
        List of documents with hybrid scores
    """
    if not _search_engine_initialized or not _bm25_retriever:
        logger.warning("⚠️ Search engine not initialized, checking if we can force initialize...")
        force_initialize_if_needed()
        
        if not _search_engine_initialized or not _bm25_retriever:
            logger.warning("⚠️ Search engine still not available, falling back to vector search only")
            return vector_search_fallback(query, limit)
    
    try:
        # 1. BM25 keyword search
        bm25_results = _bm25_retriever.retrieve(query)
        logger.info(f"🔍 BM25 search returned {len(bm25_results)} results")
        
        # 2. Vector semantic search
        vector = get_embedding(query)
        vector_results = search_vector(DEFAULT_COLLECTION_NAME, vector, limit)
        logger.info(f"🔍 Vector search returned {len(vector_results)} results")
        
        # 3. Combine and score results
        combined_results = combine_search_results(bm25_results, vector_results, query)
        
        # 4. Sort by hybrid score and limit results
        combined_results.sort(key=lambda x: x.get("hybrid_score", 0), reverse=True)
        
        logger.info(f"✅ Hybrid search returned {len(combined_results[:limit])} final results")
        return combined_results[:limit]
        
    except Exception as e:
        logger.error(f"❌ Hybrid search failed: {e}, falling back to vector search")
        return vector_search_fallback(query, limit)


def vector_search_fallback(query: str, limit: int = 5) -> List[Dict]:
    """
    Fallback to pure vector search when BM25 is not available
    
    Args:
        query: Search query
        limit: Maximum number of results
        
    Returns:
        List of documents
    """
    try:
        vector = get_embedding(query)
        results = search_vector(DEFAULT_COLLECTION_NAME, vector, limit)
        
        # Add search method metadata
        for result in results:
            result["search_method"] = "vector_fallback"
            result["hybrid_score"] = result.get("similarity_score", 0)
        
        return results
        
    except Exception as e:
        logger.error(f"Vector search fallback failed: {e}")
        return []


def combine_search_results(bm25_results, vector_results, query: str) -> List[Dict]:
    """
    Combine BM25 and vector search results using Reciprocal Rank Fusion (RRF)
    
    Args:
        bm25_results: Results from BM25 search
        vector_results: Results from vector search
        query: Original query for scoring
        
    Returns:
        List of combined documents with hybrid scores
    """
    # RRF Constant (usually 60)
    k = 60
    
    # Helper to normalize content text by removing the question if it's prepended
    def get_norm_hash(content_text: str, q: str) -> int:
        content_text = content_text.strip()
        q = q.strip()
        if q and content_text.startswith(q):
            content_text = content_text[len(q):].strip()
        return hash(content_text)

    # Convert BM25 results to dict and map doc_key -> doc
    bm25_docs = {}
    bm25_ranks = {}
    for rank_idx, node in enumerate(bm25_results, 1):
        content = node.node.text if hasattr(node.node, 'text') else str(node.node)
        question = node.node.metadata.get("question", "")
        doc_id = node.node.metadata.get("doc_id", 0)
        
        content_hash = get_norm_hash(content, question)
        doc_key = f"id_{doc_id}" if doc_id else f"hash_{content_hash}"
        
        logger.info(f"   BM25[Rank {rank_idx}]: Score={node.score:.3f}, Key={doc_key}, Q='{question[:50]}...'")
        
        bm25_docs[doc_key] = {
            "content": content,
            "question": question,
            "source": node.node.metadata.get("source", "unknown"),
            "doc_id": doc_id,
            "bm25_score": node.score,
            "search_method": "bm25"
        }
        bm25_ranks[doc_key] = rank_idx

    # Convert Vector results and map doc_key -> doc
    vector_docs = {}
    vector_ranks = {}
    logger.info(f"📝 Processing {len(vector_results)} Vector results...")
    for rank_idx, doc in enumerate(vector_results, 1):
        content = doc.get("content", "")
        question = doc.get("question", "")
        doc_id = doc.get("doc_id", 0)
        
        content_hash = get_norm_hash(content, question)
        doc_key = f"id_{doc_id}" if doc_id else f"hash_{content_hash}"
        
        logger.info(f"   Vector[Rank {rank_idx}]: Score={doc.get('similarity_score', 0):.3f}, Key={doc_key}, Q='{question[:50]}...'")
        
        vector_docs[doc_key] = {
            "content": content,
            "question": question,
            "source": doc.get("source", "unknown"),
            "doc_id": doc_id,
            "vector_score": doc.get("similarity_score", 0),
            "search_method": "vector"
        }
        vector_ranks[doc_key] = rank_idx

    # Combine results
    all_docs = {}
    overlap_count = 0
    
    logger.info(f"🔗 Combining results using RRF...")
    
    # Add BM25 results
    for doc_key, doc in bm25_docs.items():
        all_docs[doc_key] = doc
        
    # Add vector results and merge if overlap
    for doc_key, doc in vector_docs.items():
        if doc_key in all_docs:
            all_docs[doc_key]["vector_score"] = doc["vector_score"]
            all_docs[doc_key]["search_method"] = "hybrid"
            # Keep Vector's content as it's cleaner chunk content
            all_docs[doc_key]["content"] = doc["content"]
            overlap_count += 1
            logger.info(f"   ✨ Found overlap for Key {doc_key}: '{doc['question'][:40]}...' (BM25 + Vector)")
        else:
            all_docs[doc_key] = doc
            
    # Calculate RRF Scores
    w_vector = 1.15
    w_bm25 = 1.0
    hybrid_count = bm25_only_count = vector_only_count = 0
    for doc_key, doc in all_docs.items():
        rrf_score = 0.0
        
        if doc_key in bm25_ranks:
            rrf_score += w_bm25 * (1.0 / (k + bm25_ranks[doc_key]))
        if doc_key in vector_ranks:
            rrf_score += w_vector * (1.0 / (k + vector_ranks[doc_key]))
            
        doc["hybrid_score"] = rrf_score
        
        # Track counts for logging
        if doc["search_method"] == "hybrid":
            hybrid_count += 1
        elif doc["search_method"] == "bm25":
            bm25_only_count += 1
        else:
            vector_only_count += 1

    logger.info(f"   Scoring breakdown: Hybrid={hybrid_count}, BM25-only={bm25_only_count}, Vector-only={vector_only_count}")
    
    # Sort by RRF score descending
    sorted_docs = sorted(all_docs.values(), key=lambda x: x.get("hybrid_score", 0), reverse=True)
    
    logger.info(f"🏆 Top 3 combined results (RRF):")
    for i, doc in enumerate(sorted_docs[:3], 1):
        question = doc.get("question", "N/A")
        score = doc.get("hybrid_score", 0)
        method = doc.get("search_method", "unknown")
        bm25_s = doc.get("bm25_score", 0)
        vector_s = doc.get("vector_score", 0)
        logger.info(f"   {i}. {question[:50]}... (RRF Score: {score:.5f}, Method: {method}, BM25: {bm25_s:.3f}, Vec: {vector_s:.3f})")
        
    logger.info(f"✅ Combined search results: {len(all_docs)} total documents")
    return list(all_docs.values())


def search_engine() -> bool:
    """
    Alias for backward compatibility
    """
    return _search_engine_initialized


def get_search_stats() -> Dict:
    """
    Get search engine statistics
    
    Returns:
        Dict with search engine status and stats
    """
    return {
        "initialized": _search_engine_initialized,
        "has_docstore": _docstore is not None,
        "has_bm25": _bm25_retriever is not None,
        "docstore_size": len(_docstore.docs) if _docstore else 0
    }


def force_initialize_if_needed() -> bool:
    """
    Force initialize search engine if not already done
    This is a helper function to ensure search engine is ready
    """
    global _search_engine_initialized
    
    if _search_engine_initialized:
        logger.info("✅ Search engine already initialized")
        return True
        
    logger.warning("⚠️ Search engine not initialized, attempting to force initialize...")
    
    # Try to get some sample documents from the database/vector store
    try:
        from vectorize import get_collection_stats
        from config import DEFAULT_COLLECTION_NAME
        
        logger.info(f"🔍 Checking collection: {DEFAULT_COLLECTION_NAME}")
        stats = get_collection_stats(DEFAULT_COLLECTION_NAME)
        logger.info(f"📊 Collection stats: {stats}")
        
        if stats and not stats.get('error'):
            # Check both vectors_count and points_count as fallback
            vectors_count = stats.get('vectors_count') or 0
            points_count = stats.get('points_count') or 0
            
            logger.info(f"🔢 Parsed counts - vectors: {vectors_count}, points: {points_count}")
            
            if vectors_count > 0 or points_count > 0:
                logger.info(f"📊 Found {vectors_count or points_count} documents in collection (vectors: {vectors_count}, points: {points_count})")
                logger.info("🔄 Attempting to initialize search index from existing data...")
                
                # Try to initialize from existing vector data
                success = initialize_from_vector_store()
                if success:
                    logger.info("✅ Successfully initialized search index from vector store!")
                    return True
                else:
                    logger.warning("💡 Please run import_data.py to initialize the search index properly")
            else:
                logger.warning(f"📋 No documents found in collection. vectors_count={vectors_count}, points_count={points_count}")
        else:
            error_msg = stats.get('error', 'Unknown error') if stats else 'Collection not found'
            logger.warning(f"📋 Could not access collection: {error_msg}")
            
    except Exception as e:
        logger.error(f"❌ Error checking collection stats: {e}")
    
    return False


def initialize_from_vector_store(limit: int = 100000) -> bool:
    """
    Initialize search index from existing vector store data using paginated scrolls
    with native BM25 persistence caching to achieve near-instant startup (1.7s instead of 15s).
    
    Args:
        limit: Maximum number of documents to load (default: 100,000)
        
    Returns:
        bool: True if successful
    """
    try:
        from vectorize import get_client, get_collection_stats
        from config import DEFAULT_COLLECTION_NAME
        from pathlib import Path
        import json
        import os
        import time

        logger.info(f"🔄 Loading documents from vector store (limit: {limit})")

        # 1. Fetch points count from Qdrant stats to validate cache
        stats = get_collection_stats(DEFAULT_COLLECTION_NAME)
        qdrant_count = 0
        if stats and not stats.get('error'):
            qdrant_count = stats.get('points_count') or stats.get('vectors_count') or 0

        # We store native BM25 cache inside the project data directory
        cache_dir = Path("e:/MachineLearning/Legal/data/bm25_cache")
        retriever_json = cache_dir / "retriever.json"
        metadata_path = cache_dir / "cache_metadata.json"

        # 2. Try loading the native BM25 retriever directly from cache
        if retriever_json.exists() and metadata_path.exists() and qdrant_count > 0:
            try:
                # Read cached qdrant count from metadata to prevent deduplication count mismatches
                with open(metadata_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                cached_qdrant_count = meta.get("qdrant_count", 0)

                if cached_qdrant_count == qdrant_count:
                    logger.info(f"💾 Found matching BM25 native cache (Qdrant points: {qdrant_count}). Loading...")
                    start_time = time.monotonic()
                    loaded_retriever = BM25Retriever.from_persist_dir(path=str(cache_dir))
                    
                    global _bm25_retriever, _search_engine_initialized, _docstore
                    _bm25_retriever = loaded_retriever
                    _search_engine_initialized = True
                    
                    corpus_len = len(loaded_retriever.corpus) if hasattr(loaded_retriever, 'corpus') else 0
                    
                    # Use a lightweight dummy docstore to avoid heavy Python object copying
                    class DummyDocstore:
                        def __init__(self, size):
                            self.docs = range(size)
                    _docstore = DummyDocstore(corpus_len)
                    logger.info(f"✅ BM25 native cache loaded successfully in {time.monotonic() - start_time:.4f}s!")
                    return True
                else:
                    logger.info(f"⚠️ Cache count mismatch (Cache Qdrant points: {cached_qdrant_count}, Current Qdrant points: {qdrant_count}). Rebuilding index from scratch...")
            except Exception as cache_err:
                logger.warning(f"⚠️ Failed to load BM25 native cache: {cache_err}")

        # 3. Fallback: load documents from Qdrant/local cache
        cache_path = Path("e:/MachineLearning/Legal/data/search_documents_cache.json")
        documents = []
        cache_loaded = False

        if cache_path.exists() and qdrant_count > 0:
            try:
                logger.info(f"💾 Found local cache file: {cache_path}. Verifying document count...")
                with open(cache_path, "r", encoding="utf-8") as f:
                    cached_data = json.load(f)
                if isinstance(cached_data, list) and len(cached_data) == qdrant_count:
                    logger.info(f"✅ Cache count ({len(cached_data)}) matches Qdrant points ({qdrant_count}). Loading from cache...")
                    documents = cached_data
                    cache_loaded = True
                else:
                    logger.info(f"⚠️ Cache count mismatch (Cache: {len(cached_data) if isinstance(cached_data, list) else 'invalid'}, Qdrant: {qdrant_count}). Re-fetching from Qdrant...")
            except Exception as cache_err:
                logger.warning(f"⚠️ Failed to read search cache: {cache_err}")

        if not cache_loaded:
            logger.info("📡 Fetching documents from Qdrant via paginated scroll...")
            # Get documents directly from Qdrant using paginated scroll
            points = []
            next_offset = None
            client = get_client()
            
            while len(points) < limit:
                scroll_result = client.scroll(
                    collection_name=DEFAULT_COLLECTION_NAME,
                    limit=10000,
                    offset=next_offset,
                    with_payload=True,
                    with_vectors=False
                )
                batch_points, next_offset = scroll_result
                points.extend(batch_points)
                logger.info(f"   Scrolled batch: loaded {len(batch_points)} points (total: {len(points)})")
                
                if not next_offset or not batch_points:
                    break
            
            # Trim points list to limit if needed
            points = points[:limit]
            
            if not points:
                logger.warning("📋 No documents found in vector store")
                return False
            
            # Convert vector store results to the format expected by initialize_search_index
            for point in points:
                payload = point.payload
                doc = {
                    'question': payload.get('question', ''),
                    'content': payload.get('content', ''),
                    'source': payload.get('source', 'vector_store'),
                    'doc_id': payload.get('doc_id', point.id)
                }
                documents.append(doc)
                
            # Write to cache atomically
            try:
                os.makedirs(cache_path.parent, exist_ok=True)
                temp_path = cache_path.with_suffix(".tmp")
                with open(temp_path, "w", encoding="utf-8") as f:
                    json.dump(documents, f, ensure_ascii=False)
                os.replace(temp_path, cache_path)
                logger.info(f"💾 Successfully cached {len(documents)} documents to {cache_path}")
            except Exception as cache_err:
                logger.warning(f"⚠️ Failed to write search cache: {cache_err}")

        logger.info(f"📄 Loaded {len(documents)} total documents for initialization")
        logger.info(f"📝 Converted {len(documents)} documents")
        
        # Build search index
        success = initialize_search_index(documents)
        
        # Persist BM25 retriever and metadata to native cache
        if success and _bm25_retriever:
            try:
                os.makedirs(cache_dir, exist_ok=True)
                _bm25_retriever.persist(path=str(cache_dir))
                
                # Write matching metadata file
                with open(metadata_path, "w", encoding="utf-8") as f:
                    json.dump({"qdrant_count": qdrant_count}, f)
                    
                logger.info(f"💾 Successfully saved BM25 index and metadata to native cache: {cache_dir}")
            except Exception as persist_err:
                logger.warning(f"⚠️ Failed to save BM25 native cache: {persist_err}")
                
        return success
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize from vector store: {e}")
        logger.exception("Full error traceback:")
        return False