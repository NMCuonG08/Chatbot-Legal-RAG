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
from search import initialize_search_index
from splitter import split_document
from utils import setup_logging
from vectorize import add_vector, create_collection, delete_vectors_by_filter
from database import SessionLocal

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
):
    """
    Import Q&A data from JSONL file into Qdrant vector database

    Args:
        data_file_path: Path to the train.jsonl file (RAG format)
        collection_name: Name of the Qdrant collection
        batch_size: Number of vectors to process in each batch
        limit: Maximum number of records to process (None for all)
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
        f"🚀 Starting import from {data_file_path} to collection {collection_name}"
    )
    if limit:
        logger.info(f"Limiting import to {limit} records")

    # Try to create collection (will fail if already exists, which is fine)
    try:
        create_collection(collection_name)
        logger.info(f"✅ Created collection: {collection_name}")
    except Exception as e:
        logger.info(f"📋 Collection {collection_name} might already exist: {e}")

    logger.info("🔄 Starting to read JSONL file...")
    
    # Read and process the JSONL file
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
        logger.info("📄 File opened successfully, starting line-by-line processing...")
        
        for idx, line in enumerate(f):
            # Add progress logging every 50 lines
            if idx % 50 == 0:
                logger.info(f"📊 Processing line {idx + 1}...")
                
            # Check limit
            if limit and idx >= limit:
                logger.info(f"🛑 Reached limit of {limit} records, stopping")
                break

            try:
                # Parse JSON line
                data = json.loads(line.strip())
                question = data.get("question", "")
                context = data.get("context", "")  # Note: "context" thay vì "answer"

                if not question or not context:
                    logger.warning(
                        f"⚠️ Line {idx + 1}: Missing question or context, skipping"
                    )
                    error_count += 1
                    continue

                # Debug first few records
                if idx < 3:
                    logger.info(f"📝 Sample record {idx + 1}: Q='{question[:50]}...', C='{context[:50]}...'")

                # Store document for search index
                documents_for_search.append({
                    "question": question,
                    "content": context,
                    "source": "train",
                    "doc_id": idx
                })

                doc_id_str = f"train-{idx}"
                text = f"{question} {context}"
                doc_hash = hashlib.md5(text.encode("utf-8")).hexdigest()
                full_doc_cid = f"{doc_id_str}#full_doc"

                old_chunks = get_doc_chunks(doc_id_str)
                old_chunks_dict = {c.chunk_id: c.chunk_hash for c in old_chunks}

                db = SessionLocal()
                try:
                    # If this is the FIRST time indexing this document in the new system (MySQL has no chunks),
                    # we clean up any old vectors belonging to this doc_id from Qdrant to avoid duplicates.
                    if not old_chunks:
                        logger.info(f"First-time indexing for document {doc_id_str} in the new system. Cleaning old Qdrant vectors...")
                        delete_vectors_by_filter(collection_name, {"doc_id": doc_id_str})
                        try:
                            delete_vectors_by_filter(collection_name, {"doc_id": idx})
                        except Exception:
                            pass

                    # Fast-path check: If full document hash matches, skip splitting and embedding entirely!
                    if full_doc_cid in old_chunks_dict and old_chunks_dict[full_doc_cid] == doc_hash:
                        logger.info(f"Skipped entire document {doc_id_str} - full content hash matches.")
                        success_count += 1
                        continue

                    nodes = split_document(text)

                    new_chunk_ids = set()
                    to_upsert = []
                    to_delete = []

                    # Classify chunks
                    for chunk_idx, node in enumerate(nodes):
                        cid = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"doc_{doc_id_str}_chunk_{chunk_idx}"))
                        new_chunk_ids.add(cid)
                        chash = hashlib.md5(node.text.encode("utf-8")).hexdigest()

                        if cid not in old_chunks_dict or old_chunks_dict[cid] != chash:
                            to_upsert.append((cid, node.text, chash))

                    # Check deleted chunks
                    for old_cid in old_chunks_dict.keys():
                        if old_cid != full_doc_cid and old_cid not in new_chunk_ids:
                            to_delete.append(old_cid)

                    # 1. Delete orphaned chunks
                    if to_delete:
                        delete_vectors_by_ids(collection_name, to_delete)
                        delete_doc_chunks_by_ids(to_delete, db=db)

                    # 2. Embed and upsert added/modified chunks
                    if to_upsert:
                        texts_to_embed = [item[1] for item in to_upsert]
                        try:
                            embeddings = get_custom_embedding(texts_to_embed)
                            if not isinstance(embeddings, list) or (embeddings and not isinstance(embeddings[0], list)):
                                embeddings = [embeddings]
                        except Exception as e:
                            logger.error(f"Failed to generate embeddings in batch: {e}")
                            raise e

                        vectors_payload = {}
                        for (cid, text_val, chash), vector in zip(to_upsert, embeddings):
                            vectors_payload[cid] = {
                                "vector": vector,
                                "payload": {
                                    "question": question,
                                    "content": text_val,
                                    "source": "train",
                                    "doc_id": idx,
                                }
                            }
                            save_doc_chunk(doc_id_str, cid, chash, db=db)

                        add_vector(
                            collection_name=collection_name,
                            vectors=vectors_payload,
                            batch_size=batch_size,
                        )
                        total_vectors_processed += len(to_upsert)

                    # Save full document hash
                    save_doc_chunk(doc_id_str, full_doc_cid, doc_hash, db=db)

                    # Atomically commit transaction!
                    db.commit()
                    success_count += 1

                except Exception as e:
                    logger.error(f"Transaction failed for line {idx + 1}, rolling back: {e}")
                    db.rollback()
                    error_count += 1
                finally:
                    db.close()

            except json.JSONDecodeError as e:
                logger.error(f"Line {idx + 1}: JSON decode error - {e}")
                error_count += 1
            except Exception as e:
                logger.error(f"Line {idx + 1}: Unexpected error - {e}")
                error_count += 1

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

    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("Starting RAG Data Import")
    logger.info("=" * 60)
    logger.info(f"Data file: {args.data_file}")
    logger.info(f"Collection: {args.collection}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Limit: {args.limit or 'No limit'}")
    logger.info("=" * 60)

    success = import_qa_data(
        data_file_path=args.data_file,
        collection_name=args.collection,
        batch_size=args.batch_size,
        limit=args.limit,
    )

    if success:
        logger.info("Import completed successfully!")
        sys.exit(0)
    else:
        logger.error("Import failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()