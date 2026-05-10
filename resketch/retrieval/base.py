from __future__ import annotations

from typing import Protocol

from resketch.models import CandidateRequest, CandidateSet


class ProviderError(RuntimeError):
    """Raised when a candidate provider fails."""


class ProviderUnavailableError(ProviderError):
    """Raised when a configured provider is not yet available."""


class CandidateProvider(Protocol):
    def retrieve(self, request: CandidateRequest) -> CandidateSet:
        """Retrieve candidate regex components for a typed sketch hole."""
