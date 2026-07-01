"""Multi-source data pipeline for the Legal RAG system.

Architecture principle: separate *data sources* (ingestion) from *processing
logic* via a single normalized intermediate schema. Processing stages never
know where data came from — they only ever receive a :class:`RawDocument` and
branch on ``source_type``.

Adding a new source = write one connector implementing
:class:`pipeline.connectors.base.BaseConnector`. The parser / chunker /
embedder / state store are untouched.

See ``README.md`` (Data Pipeline section) for the full contract.
"""

from pipeline.connectors.base import BaseConnector
from pipeline.schema import ChunkedDocument, ParsedDocument, RawDocument

__all__ = [
    "BaseConnector",
    "ChunkedDocument",
    "ParsedDocument",
    "RawDocument",
]