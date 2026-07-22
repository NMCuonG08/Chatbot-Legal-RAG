"""
Enhanced Search Module for Vietnamese Legal Chatbot
Implements hybrid search combining semantic vector search + BM25 keyword search using LlamaIndex
"""

import hashlib
import logging
import os
import pickle
import re
import threading
from pathlib import Path
from typing import Dict, List, Optional

from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.retrievers.bm25 import BM25Retriever

from brain import get_embedding
from config import DEFAULT_COLLECTION_NAME
from vectorize import search_vector

logger = logging.getLogger(__name__)

# Vietnamese word segmentation for BM25. Vietnamese is not whitespace-tokenized
# in any meaningful way (compound words like "Đất đai", "thời hiệu" need
# segmenting to match well). pyvi (ViTokenizer) is a lightweight tokenizer that
# inserts underscores between syllables of a word, e.g. "Đất_đai" — BM25 then
# treats each segmented word as a token. Imported lazily + guarded so the
# module still loads if pyvi is not installed (degrades to raw-text BM25).
_VI_TOKENIZE_ENABLED = False
try:
    from pyvi import ViTokenizer  # type: ignore
    _VI_TOKENIZE_ENABLED = True
except Exception:  # noqa: BLE001 — optional dep
    ViTokenizer = None  # type: ignore


def _tokenize_vi(text: str) -> str:
    """Segment Vietnamese text for BM25 indexing/querying.

    Returns the original text unchanged when pyvi is unavailable so the system
    keeps working (with weaker BM25) instead of crashing. Logs once on first
    real use when the tokenizer is missing.
    """
    if not text:
        return ""
    if _VI_TOKENIZE_ENABLED and ViTokenizer is not None:
        try:
            return ViTokenizer.tokenize(text)
        except Exception as exc:  # noqa: BLE001 — never let tokenizer break search
            logger.warning("pyvi tokenize failed (%s) — using raw text", exc)
            return text
    return text


# Global search components
_docstore = None
_bm25_retriever = None
_search_engine_initialized = False

# Audit (external) Bug 2: guard the init/swap of the global _docstore /
# _bm25_retriever against concurrent cold-start in one process (e.g. Uvicorn
# threadpool firing two first queries at once). RLock so a public init entry
# point that delegates to another locked init path cannot self-deadlock.
# Celery workers are separate processes (no cross-process race), but a single
# multi-threaded process can still race the None->retriever swap.
_init_lock = threading.RLock()


def initialize_search_index(documents: List[Dict]) -> bool:
    """
    Initialize BM25 search index from documents
    
    Args:
        documents: List of documents with keys: question, content, source, doc_id
        
    Returns:
        bool: True if successful, False otherwise
    """
    global _docstore, _bm25_retriever, _search_engine_initialized

    # Audit (external) Bug 2: hold _init_lock across the whole init so two
    # concurrent cold-starts in one process cannot race the None->retriever
    # swap (one reading a half-written _bm25_retriever -> NoneType.retrieve).
    # RLock: force_initialize_if_needed -> initialize_from_vector_store can
    # re-enter without self-deadlock.
    with _init_lock:
        try:
            logger.info(f"🔧 Initializing search index with {len(documents)} documents")

            # Convert documents directly to TextNode objects to bypass slow SentenceSplitter parsing
            from llama_index.core.schema import TextNode
            nodes = []
            for i, doc in enumerate(documents):
                if not doc.get('question') and not doc.get('content'):
                    continue
                text = _tokenize_vi(f"{doc.get('question', '')} {doc.get('content', '')}")
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
        # 1. BM25 keyword search (query segmented with pyvi for Vietnamese match)
        bm25_results = _bm25_retriever.retrieve(_tokenize_vi(query))
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


def _content_norm_hash(content_text: str, q: str) -> str:
    """Stable dedup key for a chunk, used when ``doc_id`` is falsy.

    Audit (external) Bugs 1 + 3: BM25 text is pyvi-tokenized (compound words
    joined with underscores, e.g. ``"Đất_đai"``) while vector-search text is
    raw (``"Đất đai"``). A naive ``hash(content)`` produced DIFFERENT keys for
    the same chunk -> RRF saw zero overlap -> the chunk appeared twice in the
    context window. This normalizes away whitespace AND underscores (and case)
    before hashing so tokenized and raw forms collapse to one key.

    Uses md5 (deterministic across process restarts, unlike Python's
    process-randomized ``hash()`` for str) and returns a hex string. This is a
    query-time in-memory dedup key only — nothing is persisted by this hash, so
    there is no restart cache-corruption impact; the determinism is correctness
    hygiene + collision resistance.
    """
    content_text = (content_text or "").strip()
    q = (q or "").strip()
    if q and content_text.startswith(q):
        content_text = content_text[len(q):].strip()
    normalized = re.sub(r"[\s_]+", "", content_text).lower()
    return hashlib.md5(normalized.encode("utf-8")).hexdigest()


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

    # Convert BM25 results to dict and map doc_key -> doc
    bm25_docs = {}
    bm25_ranks = {}
    for rank_idx, node in enumerate(bm25_results, 1):
        content = node.node.text if hasattr(node.node, 'text') else str(node.node)
        question = node.node.metadata.get("question", "")
        doc_id = node.node.metadata.get("doc_id", 0)

        content_hash = _content_norm_hash(content, question)
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
        
        content_hash = _content_norm_hash(content, question)
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
            
    # Calculate RRF Scores. Weights are configurable via env so ops can tune
    # vector vs BM25 emphasis per deployment without a redeploy of code.
    # Defaults preserve the prior hand-tuned values (vector slightly favored).
    try:
        w_vector = float(os.environ.get("RRF_W_VECTOR", "1.15"))
    except ValueError:
        w_vector = 1.15
    try:
        w_bm25 = float(os.environ.get("RRF_W_BM25", "1.0"))
    except ValueError:
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


def _minmax(values: List[float]) -> List[float]:
    """Normalize a list of floats to [0, 1]. Returns zeros if all equal/empty."""
    if not values:
        return []
    lo = min(values)
    hi = max(values)
    if hi - lo <= 1e-12:
        return [1.0 for _ in values]
    return [(v - lo) / (hi - lo) for v in values]


def blend_hybrid_rerank(docs: List[Dict], alpha: Optional[float] = None) -> List[Dict]:
    """Blend RRF ``hybrid_score`` with reranker ``relevance_score``.

    ``final_score = alpha * norm(hybrid_score) + (1 - alpha) * norm(relevance_score)``

    Both scores live on different scales (RRF is a small reciprocal-rank sum;
    Cohere/BGE relevance is a probability-ish float), so each list is min-max
    normalized to [0, 1] before blending. Re-sort by ``final_score``.

    - ``alpha``: weight on the hybrid score; ``None`` reads ``RRF_BLEND_ALPHA``
      env (default 0.6 — favor hybrid ranking, let reranker reorder ties).
    - Docs missing ``relevance_score`` (e.g. rerank passthrough /
      ``rerank_failed=True``) fall back to their normalized hybrid score alone,
      so a reranker outage degrades gracefully instead of zeroing scores.
    """
    if alpha is None:
        try:
            alpha = float(os.environ.get("RRF_BLEND_ALPHA", "0.6"))
        except ValueError:
            alpha = 0.6
    alpha = max(0.0, min(1.0, alpha))

    if not docs:
        return []

    hybrid_scores = [float(d.get("hybrid_score", 0.0) or 0.0) for d in docs]
    rel_scores = [
        float(d["relevance_score"]) if d.get("relevance_score") is not None else None
        for d in docs
    ]
    norm_hybrid = _minmax(hybrid_scores)
    # Normalize only over docs that actually have a rerank score.
    present_rel = [v for v in rel_scores if v is not None]
    rel_lo = min(present_rel) if present_rel else 0.0
    rel_hi = max(present_rel) if present_rel else 0.0
    rel_span = (rel_hi - rel_lo) if (rel_hi - rel_lo) > 1e-12 else 1.0

    out: List[Dict] = []
    for i, doc in enumerate(docs):
        d = dict(doc)
        h = norm_hybrid[i] if i < len(norm_hybrid) else 0.0
        r = rel_scores[i]
        if r is None:
            # No rerank signal (passthrough) — keep hybrid ranking only.
            d["final_score"] = h
            d["blended"] = False
        else:
            nr = (r - rel_lo) / rel_span if rel_span > 1e-12 else 1.0
            d["final_score"] = alpha * h + (1.0 - alpha) * nr
            d["blended"] = True
        out.append(d)

    out.sort(key=lambda x: x.get("final_score", 0.0), reverse=True)
    return out


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


import threading

_rebuild_lock = threading.Lock()
_rebuilding = False

def _bg_rebuild_cache(limit: int, qdrant_count: int, cache_dir: Path, metadata_path: Path, cache_path: Path):
    global _rebuilding, _rebuild_lock, _bm25_retriever, _docstore, _search_engine_initialized
    with _rebuild_lock:
        if _rebuilding:
            return
        _rebuilding = True
        
    try:
        logger.info("⏳ [BG-REBUILD] Starting background BM25 cache rebuild...")
        from vectorize import get_client
        from config import DEFAULT_COLLECTION_NAME
        from llama_index.core.storage.docstore import SimpleDocumentStore
        from llama_index.retrievers.bm25 import BM25Retriever
        from llama_index.core.schema import TextNode
        import json
        import os
        
        # 1. Fetch points from Qdrant
        points = []
        next_offset = None
        client = get_client()
        
        while len(points) < limit:
            scroll_result = client.scroll(
                collection_name=DEFAULT_COLLECTION_NAME,
                limit=2000,
                offset=next_offset,
                with_payload=True,
                with_vectors=False
            )
            batch_points, next_offset = scroll_result
            points.extend(batch_points)
            if not next_offset or not batch_points:
                break
                
        points = points[:limit]
        if not points:
            logger.warning("[BG-REBUILD] No points fetched from Qdrant")
            return
            
        documents = []
        for point in points:
            payload = point.payload
            doc = {
                'question': payload.get('question', ''),
                'content': payload.get('content', ''),
                'source': payload.get('source', 'vector_store'),
                'doc_id': payload.get('doc_id', point.id)
            }
            documents.append(doc)
            
        # 2. Write search_documents_cache.json
        os.makedirs(cache_path.parent, exist_ok=True)
        temp_path = cache_path.with_suffix(".tmp")
        with open(temp_path, "w", encoding="utf-8") as f:
            json.dump(documents, f, ensure_ascii=False)
        try:
            os.replace(temp_path, cache_path)
        except PermissionError:
            pass # Windows lock, ignore since content matches
            
        # 3. Build nodes and docstore
        nodes = []
        for i, doc in enumerate(documents):
            if not doc.get('question') and not doc.get('content'):
                continue
            text = _tokenize_vi(f"{doc.get('question', '')} {doc.get('content', '')}")
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
            
        new_docstore = SimpleDocumentStore()
        new_docstore.add_documents(nodes)
        
        new_retriever = BM25Retriever.from_defaults(
            docstore=new_docstore,
            similarity_top_k=5,
        )
        
        # 4. Persist BM25 native cache
        os.makedirs(cache_dir, exist_ok=True)
        new_retriever.persist(path=str(cache_dir))
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump({"qdrant_count": qdrant_count}, f)
            
        # 5. Swap globals
        _docstore = new_docstore
        _bm25_retriever = new_retriever
        _search_engine_initialized = True
        logger.info(f"✅ [BG-REBUILD] Successfully rebuilt and swapped BM25 retriever in memory with {len(documents)} docs!")
    except Exception as rebuild_err:
        logger.error(f"❌ [BG-REBUILD] Background rebuild error: {rebuild_err}")
    finally:
        with _rebuild_lock:
            _rebuilding = False


def initialize_from_vector_store(limit: int = 300000) -> bool:
    """
    Initialize search index from existing vector store data using paginated scrolls
    with native BM25 persistence caching to achieve near-instant startup (1.7s instead of 15s).
    
    Args:
        limit: Maximum number of documents to load (default: 300,000)
        
    Returns:
        bool: True if successful
    """
    global _bm25_retriever, _search_engine_initialized, _docstore
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
                    logger.info(f"⚠️ Cache count mismatch (Cache Qdrant points: {cached_qdrant_count}, Current Qdrant points: {qdrant_count}). Loading stale cache fallback and rebuilding in background...")
                    start_time = time.monotonic()
                    loaded_retriever = BM25Retriever.from_persist_dir(path=str(cache_dir))
                    
                    _bm25_retriever = loaded_retriever
                    _search_engine_initialized = True
                    
                    corpus_len = len(loaded_retriever.corpus) if hasattr(loaded_retriever, 'corpus') else 0
                    
                    class DummyDocstore:
                        def __init__(self, size):
                            self.docs = range(size)
                    _docstore = DummyDocstore(corpus_len)
                    logger.info(f"✅ Stale BM25 native cache loaded successfully in {time.monotonic() - start_time:.4f}s! Spawning background rebuild...")
                    
                    # Spawn background rebuild thread
                    cache_path = Path("e:/MachineLearning/Legal/data/search_documents_cache.json")
                    t = threading.Thread(
                        target=_bg_rebuild_cache,
                        args=(limit, qdrant_count, cache_dir, metadata_path, cache_path),
                        daemon=True
                    )
                    t.start()
                    return True
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
                expected_count = min(limit, qdrant_count)
                if isinstance(cached_data, list) and len(cached_data) == expected_count:
                    logger.info(f"✅ Cache count ({len(cached_data)}) matches expected points ({expected_count}). Loading from cache...")
                    documents = cached_data
                    cache_loaded = True
                else:
                    logger.info(f"⚠️ Cache count mismatch (Cache: {len(cached_data) if isinstance(cached_data, list) else 'invalid'}, Expected: {expected_count}). Re-fetching from Qdrant...")
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
                    limit=2000,
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