"""
Evidence-grounded, model-graded treatment recommendations (P2.3).

The disease playbooks in :mod:`CrewAI.orchestrator.services` select
*candidate* recommendations by ``(disease, positive, risk_level)``. This
module turns candidates into recommendations a clinician can trust:

- **Evidence-grounded**: each candidate is scored against the retrieved
  evidence chunks by content-token overlap, and recommendations that
  clear ``MIN_CITATION_SUPPORT`` carry inline ``[evidence: doc_id]``
  citations. Candidates with no retrieved support are kept (dropping
  them could leave an empty plan) but labeled ``[playbook-only: no
  retrieved evidence]`` so unsupported advice is never presented as
  evidence-based.
- **Model-graded**: candidates mentioning the patient's model-derived
  drivers (e.g. top SHAP features from :func:`attribute_tabular`) rank
  higher, so the plan leads with what actually drove this patient's
  prediction rather than generic playbook order.

All scoring is deterministic token overlap — no LLM, no new
dependency. Callers with no evidence or drivers keep the legacy
playbook order (see :func:`build_treatment_recommendations`).
"""

from __future__ import annotations

import re

from collections.abc import Sequence
from dataclasses import dataclass

from .schemas import EvidenceItem

#: Weight of retrieved-evidence support in the final grade.
W_EVIDENCE = 0.6

#: Weight of patient-driver relevance in the final grade.
W_DRIVER = 0.3

#: Weight of the playbook's own ordering (stability on ties).
W_ORDER = 0.1

#: Minimum token-overlap support for an evidence citation — roughly one
#: shared content term in a typical-length recommendation. Higher would
#: silence citations against terse corpus chunks (a single distinctive
#: term like "metformin" is legitimate traceability); lower would cite
#: on boilerplate. Candidates below this stay labeled playbook-only.
MIN_CITATION_SUPPORT = 0.08

#: Function words excluded from overlap scoring; they co-occur in every
#: clinical sentence and would inflate support without meaning.
_STOPWORDS = frozenset(
    [
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "been",
        "by",
        "can",
        "did",
        "do",
        "does",
        "for",
        "from",
        "had",
        "has",
        "have",
        "in",
        "into",
        "is",
        "it",
        "its",
        "of",
        "on",
        "or",
        "per",
        "that",
        "the",
        "their",
        "them",
        "there",
        "these",
        "they",
        "this",
        "those",
        "to",
        "was",
        "were",
        "will",
        "with",
        "you",
        "your",
        "all",
        "any",
        "each",
        "other",
        "some",
        "such",
        "only",
        "own",
        "same",
        "than",
        "then",
        "when",
        "while",
        "about",
        "after",
        "before",
        "between",
        "through",
        "during",
        "without",
        "within",
        "should",
        "would",
        "could",
        "may",
        "might",
        "must",
        "shall",
    ]
)

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> frozenset[str]:
    """Lowercase alphanumeric tokens minus stopwords and short fragments."""
    return frozenset(
        token
        for token in _TOKEN_RE.findall(text.lower())
        if token not in _STOPWORDS and len(token) > 2
    )


def _driver_tokens(drivers: Sequence[str]) -> frozenset[str]:
    """Split driver feature names (``fasting_glucose``) into match tokens."""
    tokens: set[str] = set()
    for driver in drivers:
        tokens.update(_TOKEN_RE.findall(str(driver).lower().replace("_", " ")))
    return frozenset(token for token in tokens if len(token) > 2)


@dataclass(frozen=True)
class GradedRecommendation:
    """
    A playbook candidate scored against evidence and patient drivers.

    Parameters
    ----------
    text : str
        The playbook candidate string (without annotations).
    grade : float
        Combined score in ``[0, 1]``; higher ranks first.
    evidence_support : float
        Best chunk token-overlap in ``[0, 1]``.
    evidence_ids : tuple[str, ...]
        Cited chunk ids (overlap >= ``MIN_CITATION_SUPPORT``).
    driver_hits : tuple[str, ...]
        Driver tokens mentioned by the candidate.
    """

    text: str
    grade: float = 0.0
    evidence_support: float = 0.0
    evidence_ids: tuple[str, ...] = ()
    driver_hits: tuple[str, ...] = ()

    @property
    def supported(self) -> bool:
        """True when at least one evidence chunk substantively supports."""
        return bool(self.evidence_ids)

    def annotated(self) -> str:
        """Render with an inline citation or a playbook-only label."""
        if self.evidence_ids:
            return f"{self.text} [evidence: {', '.join(self.evidence_ids)}]"
        return f"{self.text} [playbook-only: no retrieved evidence]"


def _chunk_overlap(candidate_tokens: frozenset[str], chunk: EvidenceItem) -> float:
    """Fraction of candidate tokens appearing in the chunk text."""
    if not candidate_tokens:
        return 0.0
    chunk_tokens = _tokens(chunk.text)
    if not chunk_tokens:
        return 0.0
    return len(candidate_tokens & chunk_tokens) / len(candidate_tokens)


def grade_recommendations(
    candidates: Sequence[str],
    evidence: Sequence[EvidenceItem],
    drivers: Sequence[str] | None = None,
) -> list[GradedRecommendation]:
    """
    Score playbook candidates against evidence and patient drivers.

    Parameters
    ----------
    candidates : Sequence[str]
        Playbook strings for the patient's ``(disease, positive,
        risk_level)`` slot, in playbook order.
    evidence : Sequence[EvidenceItem]
        Retrieved chunks grounding the citations.
    drivers : Sequence[str] | None
        Model-derived driver feature names (e.g. top SHAP features);
        None skips the driver term.

    Returns
    -------
    list[GradedRecommendation]
        Graded candidates sorted by descending grade (stable: playbook
        order breaks ties).
    """

    items = list(candidates)
    driver_set = _driver_tokens(drivers or [])
    graded: list[GradedRecommendation] = []
    for index, text in enumerate(items):
        candidate_tokens = _tokens(text)
        best = 0.0
        cited: list[str] = []
        for chunk in evidence:
            overlap = _chunk_overlap(candidate_tokens, chunk)
            best = max(best, overlap)
            if overlap >= MIN_CITATION_SUPPORT and chunk.document_id:
                cited.append(chunk.document_id)
        hits = tuple(sorted(driver_set & candidate_tokens))
        driver_term = (len(hits) / len(driver_set)) if driver_set else 0.0
        order_term = 1.0 - (index / len(items)) if items else 0.0
        grade = W_EVIDENCE * best + W_DRIVER * driver_term + W_ORDER * order_term
        graded.append(
            GradedRecommendation(
                text=text,
                grade=grade,
                evidence_support=best,
                evidence_ids=tuple(cited),
                driver_hits=hits,
            )
        )
    graded.sort(key=lambda item: item.grade, reverse=True)
    return graded


__all__ = [
    "MIN_CITATION_SUPPORT",
    "W_DRIVER",
    "W_EVIDENCE",
    "W_ORDER",
    "GradedRecommendation",
    "grade_recommendations",
]
