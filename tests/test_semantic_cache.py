from unittest.mock import MagicMock, patch
import pytest
import vectorize
from backend.src.semantic_cache import (
    init_semantic_cache,
    get_cached_response,
    set_cached_response,
    CACHE_COLLECTION_NAME
)


def test_semantic_cache_flow():
    # Inject a mock Qdrant client via the DI seam (no live Qdrant needed).
    mock_client = MagicMock()

    # Mock client.get_collections().collections to return list of mock collections
    mock_collections_holder = MagicMock()
    mock_collection_item = MagicMock()
    mock_collection_item.name = CACHE_COLLECTION_NAME
    mock_collections_holder.collections = [mock_collection_item]
    mock_client.get_collections.return_value = mock_collections_holder

    # Mock search response
    mock_point = MagicMock()
    mock_point.score = 0.98
    mock_point.payload = {
        "query": "Hỏi về tuổi nghỉ hưu của lao động nam năm 2026?",
        "response": "Tuổi nghỉ hưu của nam năm 2026 là 61 tuổi 6 tháng.",
        "sources": [{"source": "Luật lao động", "doc_id": 99, "content": "Tuổi nghỉ hưu..."}]
    }

    # Patch the embedding function; inject mock Qdrant client via the DI seam.
    with patch("backend.src.semantic_cache.get_embedding", return_value=[1.0] * 1024), \
         patch.object(vectorize, "client", mock_client):

        # Initialize cache (collection already exists in mock)
        init_semantic_cache()
        mock_client.get_collections.assert_called_once()
        mock_client.create_collection.assert_not_called()

        # If collection does not exist
        mock_client.reset_mock()
        mock_collections_holder.collections = []
        init_semantic_cache()
        mock_client.get_collections.assert_called_once()
        mock_client.create_collection.assert_called_once()

        # 2. Test get_cached_response when HIT
        mock_client.search.return_value = [mock_point]
        res = get_cached_response("Hỏi về tuổi nghỉ hưu?")
        assert res is not None
        assert res["response"] == "Tuổi nghỉ hưu của nam năm 2026 là 61 tuổi 6 tháng."
        assert res["sources"] == [{"source": "Luật lao động", "doc_id": 99, "content": "Tuổi nghỉ hưu..."}]
        mock_client.search.assert_called_once()

        # 3. Test get_cached_response when MISS
        mock_client.reset_mock()
        mock_client.search.return_value = []
        res_miss = get_cached_response("Hỏi về tuổi nghỉ hưu?")
        assert res_miss is None
        mock_client.search.assert_called_once()

        # 4. Test set_cached_response
        mock_client.reset_mock()
        set_cached_response(
            "Hỏi về tuổi nghỉ hưu?",
            "Tuổi nghỉ hưu của nam năm 2026 là 61 tuổi 6 tháng.",
            [{"source": "Luật lao động", "doc_id": 99, "content": "Tuổi nghỉ hưu..."}]
        )
        mock_client.upsert.assert_called_once()
