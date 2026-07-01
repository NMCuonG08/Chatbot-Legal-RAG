"""Base connector interface — every data source implements this."""

from __future__ import annotations

from abc import ABC, abstractmethod

from pipeline.schema import RawDocument


class BaseConnector(ABC):
    """One connector per data source, sharing one interface.

    Subclasses set ``source_id`` (stable short id used in lineage + the state
    store) and ``source_type`` (drives parser dispatch in
    :mod:`pipeline.parsers`).
    """

    source_id: str = "base"
    source_type: str = "generic"

    @abstractmethod
    def fetch(self) -> list[RawDocument]:
        """Fetch data from the source, return normalized :class:`RawDocument` list."""
        raise NotImplementedError