"""
Script to import data from train_qa_format.jsonl into Qdrant vector database
"""

import json
import logging
import os
import sys
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent))

from brain import get_embedding
from config import DEFAULT_COLLECTION_NAME
from legal_metadata import extract_legal_metadata
from legal_graph_ingest import add_to_graph
from legal_effectivity import effectivity_for_payload
from search import initialize_search_index
from splitter import split_document
from utils import setup_logging
from vectorize import add_vector, create_collection, delete_vectors_by_filter
from database import SessionLocal
from sqlalchemy import delete

setup_logging()
logger = logging.getLogger(__name__)

# Path to the data file
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LOCAL_DATA_FILE_PATH = REPO_ROOT / "data" / "train.jsonl"
DEFAULT_CONTAINER_DATA_FILE_PATH = Path("/usr/src/app/data/train.jsonl")


def _resolve_default_data_file_path() -> str:
    env_path = os.getenv("IMPORT_DATA_FILE_PATH")
    if env_path:
        return env_path

    if DEFAULT_CONTAINER_DATA_FILE_PATH.exists():
        return str(DEFAULT_CONTAINER_DATA_FILE_PATH)

    return str(DEFAULT_LOCAL_DATA_FILE_PATH)


DATA_FILE_PATH = _resolve_default_data_file_path()


def import_qa_data(
    data_file_path=DATA_FILE_PATH,
    collection_name=DEFAULT_COLLECTION_NAME,
    batch_size=50,
    limit=None,
    reset=False,
):
    """
    Import Q&A data from JSONL file into Qdrant vector database

    Args:
        data_file_path: Path to the train.jsonl file (RAG format)
        collection_name: Name of the Qdrant collection
        batch_size: Number of vectors to process in each batch
        limit: Maximum number of records to process (None for all)
        reset: Reset database and start fresh (deletes MySQL chunks and Qdrant collection)
    """

    # Fall back to defaults when callers pass null/None values from Swagger.
    data_file_path = data_file_path or DATA_FILE_PATH
    collection_name = collection_name or DEFAULT_COLLECTION_NAME
    batch_size = batch_size or 50

    # Check if file exists
    if not os.path.exists(data_file_path):
        logger.error(f"❌ Data file not found: {data_file_path}")
        return False

    logger.info(f"✅ Data file found: {data_file_path}")
    
    # Get file size for progress tracking
    file_size = os.path.getsize(data_file_path)
    logger.info(f"📊 File size: {file_size / (1024*1024):.2f} MB")

    logger.info(
        f"🚀 Starting import from {data_file_path} to collection {collection_name} (reset={reset})"
    )
    # Clear and recreate collection to start fresh
    from vectorize import get_client
    from models import DocumentChunk
    
    if reset:
        # 1. Clear MySQL document_chunks table
        db_clear = SessionLocal()
        try:
            logger.info("🗑️ Deleting all records from MySQL document_chunks table to reset incremental index state...")
            db_clear.execute(delete(DocumentChunk))
            db_clear.commit()
            logger.info("✅ MySQL document_chunks table cleared successfully")
        except Exception as e:
            db_clear.rollback()
            logger.error(f"❌ Failed to clear MySQL document_chunks table: {e}")
        finally:
            db_clear.close()

        # 2. Recreate Qdrant collection
        try:
            logger.info(f"🗑️ Deleting existing Qdrant collection {collection_name} to clear old vector data...")
            get_client().delete_collection(collection_name)
        except Exception as e:
            logger.warning(f"📋 Failed to delete Qdrant collection {collection_name} (it may not exist yet): {e}")

        try:
            create_collection(collection_name)
            logger.info(f"✅ Created fresh Qdrant collection: {collection_name}")
        except Exception as e:
            logger.error(f"❌ Failed to create collection {collection_name}: {e}")
    else:
        # Resume mode - only create collection if it doesn't exist
        try:
            get_client().get_collection(collection_name)
            logger.info(f"🔄 Collection {collection_name} exists, resuming indexing in incremental mode.")
        except Exception:
            try:
                create_collection(collection_name)
                logger.info(f"✅ Collection {collection_name} didn't exist. Created fresh collection.")
            except Exception as e:
                logger.error(f"❌ Failed to create collection {collection_name}: {e}")

    logger.info("🔄 Starting to read JSONL file...")
    success_count = 0
    error_count = 0
    total_vectors_processed = 0  # Track total vectors processed
    documents_for_search = []  # Collect documents for search index

    import hashlib
    import uuid
    from models import get_doc_chunks, save_doc_chunk, delete_doc_chunks_by_ids
    from custom_embedding import get_custom_embedding
    from vectorize import delete_vectors_by_ids

    logger.info(f"📖 Opening file: {data_file_path}")
    
    with open(data_file_path, "r", encoding="utf-8") as f:
        logger.info("📄 File opened successfully, starting batch processing...")
        
        batch_lines = []
        for idx, line in enumerate(f):
            # Check limit
            if limit and idx >= limit:
                logger.info(f"🛑 Reached limit of {limit} records, stopping")
                break
                
            batch_lines.append((idx, line))
            
            if len(batch_lines) >= 50:
                logger.info(f"📊 Processing batch of documents up to index {idx + 1}...")
                
                db = SessionLocal()
                try:
                    to_upsert_global = []
                    to_delete_global = []
                    docs_to_save_hash = []
                    
                    for doc_idx, doc_line in batch_lines:
                        try:
                            # Parse JSON line
                            data = json.loads(doc_line.strip())
                            question = data.get("question", "")
                            context = data.get("context", "") or data.get("answer", "")
                            if not question or not context:
                                error_count += 1
                                continue
                                
                            # Store document for search index
                            documents_for_search.append({
                                "question": question,
                                "content": context,
                                "source": "train",
                                "doc_id": doc_idx
                            })
                            
                            doc_id_str = f"train-{doc_idx}"
                            text = f"{question} {context}"
                            doc_hash = hashlib.md5(text.encode("utf-8")).hexdigest()
                            full_doc_cid = f"{doc_id_str}#full_doc"
                            
                            old_chunks = get_doc_chunks(doc_id_str)
                            old_chunks_dict = {c.chunk_id: c.chunk_hash for c in old_chunks}
                            
                            # Clean up old vectors if first time in new system (Bypassed: collection created fresh at startup)
                            pass
                                    
                            if full_doc_cid in old_chunks_dict and old_chunks_dict[full_doc_cid] == doc_hash:
                                success_count += 1
                                continue
                                
                            nodes = split_document(text, use_semantic=False)
                            new_chunk_ids = set()
                            
                            for chunk_idx, node in enumerate(nodes):
                                cid = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"doc_{doc_id_str}_chunk_{chunk_idx}"))
                                new_chunk_ids.add(cid)
                                chash = hashlib.md5(node.text.encode("utf-8")).hexdigest()
                                
                                if cid not in old_chunks_dict or old_chunks_dict[cid] != chash:
                                    to_upsert_global.append((cid, node.text, chash, doc_id_str, doc_idx, question))
                                    
                            for old_cid in old_chunks_dict.keys():
                                if old_cid != full_doc_cid and old_cid not in new_chunk_ids:
                                    to_delete_global.append(old_cid)
                                    
                            docs_to_save_hash.append((doc_id_str, full_doc_cid, doc_hash))
                        except json.JSONDecodeError as je:
                            logger.error(f"Line {doc_idx + 1}: JSON decode error - {je}")
                            error_count += 1
                        except Exception as de:
                            logger.error(f"Line {doc_idx + 1}: Error parsing doc - {de}")
                            error_count += 1
                            
                    # 1. Delete orphaned chunks in batch
                    if to_delete_global:
                        delete_vectors_by_ids(collection_name, to_delete_global)
                        delete_doc_chunks_by_ids(to_delete_global, db=db)
                        
                    # 2. Embed and upsert added/modified chunks in batch
                    if to_upsert_global:
                        texts_to_embed = [item[1] for item in to_upsert_global]
                        try:
                            embeddings = get_custom_embedding(texts_to_embed)
                            if not isinstance(embeddings, list) or (embeddings and not isinstance(embeddings[0], list)):
                                embeddings = [embeddings]
                        except Exception as e:
                            logger.error(f"Failed to generate embeddings in batch: {e}")
                            raise e
                            
                        vectors_payload = {}
                        for (cid, text_val, chash, doc_id_str, doc_idx, question), vector in zip(to_upsert_global, embeddings):
                            payload = {
                                "question": question,
                                "content": text_val,
                                "source": "train",
                                "doc_id": doc_idx,
                            }
                            # Enrich with structured law_name/article_number for exact-field
                            # retrieval. Only recognized keys are added (no null pollution).
                            payload.update(extract_legal_metadata(f"{question} {text_val}"))
                            # Classify effectivity from the parsed law_name/document_year so
                            # retrieval can filter out repealed/not-yet-effective statutes.
                            payload["effectivity_status"] = effectivity_for_payload(
                                payload.get("law_name"),
                                payload.get("document_year"),
                            )
                            # Phase 3 — write Statute->Article to Neo4j graph memory.
                            # Best-effort + idempotent (MERGE): no-op when graph
                            # is down/absent, never blocks vector ingest.
                            add_to_graph(cid, text_val, payload)
                            vectors_payload[cid] = {
                                "vector": vector,
                                "payload": payload,
                            }
                            save_doc_chunk(doc_id_str, cid, chash, db=db)

                        add_vector(
                            collection_name=collection_name,
                            vectors=vectors_payload,
                            batch_size=50,
                        )
                        total_vectors_processed += len(to_upsert_global)
                        
                    # 3. Save full document hashes
                    for doc_id_str, full_doc_cid, doc_hash in docs_to_save_hash:
                        save_doc_chunk(doc_id_str, full_doc_cid, doc_hash, db=db)
                        
                    db.commit()
                    success_count += len(batch_lines)
                except Exception as e:
                    db.rollback()
                    logger.error(f"Batch processing failed: {e}")
                finally:
                    db.close()
                # Respect Cohere free Trial API rate limit (10 RPM) only if we actually generated embeddings
                if to_upsert_global:
                    import time
                    keys_str = os.environ.get("COHERE_API_KEYS", "")
                    num_keys = len([k for k in keys_str.split(",") if k.strip()]) if keys_str else 1
                    sleep_time = max(0.5, 6.0 / num_keys)
                    logger.info(f"⏱️ Dynamic sleep: sleeping {sleep_time:.2f}s (configured keys: {num_keys})")
                    time.sleep(sleep_time)
                    
                batch_lines = []

        # Process remaining lines at the end
        if batch_lines:
            logger.info(f"📊 Processing final batch of documents...")
            db = SessionLocal()
            try:
                to_upsert_global = []
                to_delete_global = []
                docs_to_save_hash = []
                
                for doc_idx, doc_line in batch_lines:
                    try:
                        data = json.loads(doc_line.strip())
                        question = data.get("question", "")
                        context = data.get("context", "") or data.get("answer", "")
                        if not question or not context:
                            error_count += 1
                            continue
                            
                        documents_for_search.append({
                            "question": question,
                            "content": context,
                            "source": "train",
                            "doc_id": doc_idx
                        })
                        
                        doc_id_str = f"train-{doc_idx}"
                        text = f"{question} {context}"
                        doc_hash = hashlib.md5(text.encode("utf-8")).hexdigest()
                        full_doc_cid = f"{doc_id_str}#full_doc"
                        
                        old_chunks = get_doc_chunks(doc_id_str)
                        old_chunks_dict = {c.chunk_id: c.chunk_hash for c in old_chunks}
                        
                        # Clean up old vectors if first time in new system (Bypassed: collection created fresh at startup)
                        pass
                                
                        if full_doc_cid in old_chunks_dict and old_chunks_dict[full_doc_cid] == doc_hash:
                            success_count += 1
                            continue
                            
                        nodes = split_document(text, use_semantic=False)
                        new_chunk_ids = set()
                        
                        for chunk_idx, node in enumerate(nodes):
                            cid = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"doc_{doc_id_str}_chunk_{chunk_idx}"))
                            new_chunk_ids.add(cid)
                            chash = hashlib.md5(node.text.encode("utf-8")).hexdigest()
                            
                            if cid not in old_chunks_dict or old_chunks_dict[cid] != chash:
                                to_upsert_global.append((cid, node.text, chash, doc_id_str, doc_idx, question))
                                
                        for old_cid in old_chunks_dict.keys():
                            if old_cid != full_doc_cid and old_cid not in new_chunk_ids:
                                to_delete_global.append(old_cid)
                                
                        docs_to_save_hash.append((doc_id_str, full_doc_cid, doc_hash))
                    except json.JSONDecodeError as je:
                        logger.error(f"Line {doc_idx + 1}: JSON decode error - {je}")
                        error_count += 1
                    except Exception as de:
                        logger.error(f"Line {doc_idx + 1}: Error parsing doc - {de}")
                        error_count += 1
                        
                if to_delete_global:
                    delete_vectors_by_ids(collection_name, to_delete_global)
                    delete_doc_chunks_by_ids(to_delete_global, db=db)
                    
                if to_upsert_global:
                    texts_to_embed = [item[1] for item in to_upsert_global]
                    try:
                        embeddings = get_custom_embedding(texts_to_embed)
                        if not isinstance(embeddings, list) or (embeddings and not isinstance(embeddings[0], list)):
                            embeddings = [embeddings]
                    except Exception as e:
                        logger.error(f"Failed to generate embeddings in batch: {e}")
                        raise e
                        
                    vectors_payload = {}
                    for (cid, text_val, chash, doc_id_str, doc_idx, question), vector in zip(to_upsert_global, embeddings):
                        payload = {
                            "question": question,
                            "content": text_val,
                            "source": "train",
                            "doc_id": doc_idx,
                        }
                        # Enrich with structured law_name/article_number for exact-field
                        # retrieval. Only recognized keys are added (no null pollution).
                        payload.update(extract_legal_metadata(f"{question} {text_val}"))
                        payload["effectivity_status"] = effectivity_for_payload(
                            payload.get("law_name"),
                            payload.get("document_year"),
                        )
                        # Phase 3 — write Statute->Article to Neo4j graph memory.
                        # Best-effort + idempotent (MERGE): no-op when graph is
                        # down/absent, never blocks vector ingest.
                        add_to_graph(cid, text_val, payload)
                        vectors_payload[cid] = {
                            "vector": vector,
                            "payload": payload,
                        }
                        save_doc_chunk(doc_id_str, cid, chash, db=db)
                        
                    add_vector(
                        collection_name=collection_name,
                        vectors=vectors_payload,
                        batch_size=50,
                    )
                    total_vectors_processed += len(to_upsert_global)
                    
                for doc_id_str, full_doc_cid, doc_hash in docs_to_save_hash:
                    save_doc_chunk(doc_id_str, full_doc_cid, doc_hash, db=db)
                    
                db.commit()
                success_count += len(batch_lines)
            except Exception as e:
                db.rollback()
                logger.error(f"Final batch processing failed: {e}")
            finally:
                db.close()

    logger.info("🎯 REACHED END OF PROCESSING LOOP!")
    logger.info(
        f"📊 Import completed! Total vectors: {total_vectors_processed}, Records: {success_count}, Errors: {error_count}"
    )

    # Initialize search index with collected documents
    if documents_for_search:
        logger.info(f"🔍 STARTING SEARCH INDEX INITIALIZATION with {len(documents_for_search)} documents...")
        try:
            success = initialize_search_index(documents_for_search)
            if success:
                logger.info("✅ Search index initialized successfully!")
            else:
                logger.error("❌ Search index initialization returned False")
        except Exception as e:
            logger.error(f"❌ Failed to initialize search index: {e}")
            logger.exception("Full error traceback:")
    else:
        logger.warning("⚠️ No documents collected for search index!")

    logger.info("🏁 IMPORT FUNCTION COMPLETED!")

    # On a full reset/reindex, cached answers were grounded in the OLD corpus
    # and are now stale. Wipe the semantic cache so subsequent queries recompute
    # against the freshly indexed documents. Best-effort, never blocks the
    # successful import. Covers BOTH the CLI (main) and the API endpoint path
    # (which calls import_qa_data directly and bypasses main).
    if reset:
        try:
            from semantic_cache import clear_semantic_cache
            wiped = clear_semantic_cache()
            logger.info(f"🧹 Semantic cache wiped after reset reindex: {wiped} stale entries removed.")
        except Exception as cache_err:
            logger.warning(f"⚠️ Could not wipe semantic cache after reset: {cache_err}")

    return True


def main():
    """Main function to run the import"""
    import argparse

    parser = argparse.ArgumentParser(description="Import Q&A data into Qdrant")
    parser.add_argument(
        "--data-file",
        type=str,
        default=DATA_FILE_PATH,
        help="Path to train.jsonl file (RAG format)",
    )
    parser.add_argument(
        "--collection",
        type=str,
        default=DEFAULT_COLLECTION_NAME,
        help="Qdrant collection name",
    )
    parser.add_argument(
        "--batch-size", 
        type=int, 
        default=50, 
        help="Batch size for vector processing"
    )
    parser.add_argument(
        "--limit", 
        type=int, 
        default=None, 
        help="Limit number of records to process (for testing)"
    )
    parser.add_argument(
        "--reset", 
        action="store_true",
        help="Reset database and start fresh (deletes old data)"
    )

    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("Starting RAG Data Import")
    logger.info("=" * 60)
    logger.info(f"Data file: {args.data_file}")
    logger.info(f"Collection: {args.collection}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Limit: {args.limit or 'No limit'}")
    logger.info(f"Reset database: {args.reset}")
    logger.info("=" * 60)

    success = import_qa_data(
        data_file_path=args.data_file,
        collection_name=args.collection,
        batch_size=args.batch_size,
        limit=args.limit,
        reset=args.reset,
    )

    if success:
        logger.info("Import completed successfully!")
        # Wipe the semantic cache after a successful reindex: cached answers
        # were grounded in the OLD document set and may now be stale/wrong
        # against the freshly indexed corpus. Best-effort — never block import
        # on cache cleanup.
        try:
            from semantic_cache import clear_semantic_cache
            wiped = clear_semantic_cache()
            logger.info(f"Semantic cache wiped after reindex: {wiped} stale entries removed.")
        except Exception as cache_err:
            logger.warning(f"Could not wipe semantic cache after reindex: {cache_err}")
        sys.exit(0)
    else:
        logger.error("Import failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()