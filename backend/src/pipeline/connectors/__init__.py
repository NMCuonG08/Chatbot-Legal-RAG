"""Connector registry.

One class per data source, all implementing :class:`BaseConnector`.
Add a source = add a module here and one line in the orchestrator's
``CONNECTORS`` list. Do not branch the pipeline on the source.
"""

from pipeline.connectors.base import BaseConnector
from pipeline.connectors.html_connector import HtmlConnector
from pipeline.connectors.jsonl_qa import JsonlQaConnector
from pipeline.connectors.markdown_connector import MarkdownConnector
from pipeline.connectors.pdf_connector import PdfConnector

__all__ = [
    "BaseConnector",
    "HtmlConnector",
    "JsonlQaConnector",
    "MarkdownConnector",
    "PdfConnector",
]