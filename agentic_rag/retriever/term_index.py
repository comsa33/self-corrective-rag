"""Terminology mapping index for vocabulary gap bridging.

Maps user-facing terms to document-internal terminology, enabling
RLM tools to translate user language into document language for
more effective retrieval. No LLM calls — frequency-based extraction
with fuzzy matching.
"""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agentic_rag.retriever.indexer import Passage


class TermIndex:
    """Maps user terms to document-specific terminology.

    Built from passage titles and content during indexing.
    Uses frequency-based term extraction and substring/edit-distance
    matching to bridge vocabulary gaps between user queries and documents.
    """

    def __init__(self) -> None:
        # normalized_term → set of original surface forms
        self.term_map: dict[str, set[str]] = defaultdict(set)
        # term → document frequency (number of passages containing it)
        self.term_freq: Counter = Counter()
        # All unique terms for matching
        self._all_terms: set[str] = set()

    def build(self, passages: list[Passage]) -> None:
        """Build the terminology index from passages.

        Extracts terms from titles (high weight) and content,
        tracking document frequency for relevance ranking.
        """
        self.term_map.clear()
        self.term_freq.clear()
        self._all_terms.clear()

        for p in passages:
            # Extract terms from title (higher signal)
            title_terms = self._extract_terms(p.title)
            for term in title_terms:
                normalized = term.lower()
                self.term_map[normalized].add(term)
                self._all_terms.add(term)

            # Extract terms from content
            content_terms = self._extract_terms(p.content)
            passage_terms = set(t.lower() for t in title_terms + content_terms)
            for t in passage_terms:
                self.term_freq[t] += 1

            for term in content_terms:
                normalized = term.lower()
                self.term_map[normalized].add(term)
                self._all_terms.add(term)

    def lookup(self, user_term: str, top_k: int = 5) -> list[str]:
        """Find document terms matching a user term.

        Uses a multi-strategy approach:
        1. Exact match (normalized)
        2. Substring match (user_term in doc_term or vice versa)
        3. Common prefix match (≥3 chars)

        Returns:
            List of matching document terms, ranked by document frequency.
        """
        normalized = user_term.lower().strip()
        if not normalized:
            return []

        candidates: dict[str, float] = {}

        # Strategy 1: Exact match
        if normalized in self.term_map:
            for surface in self.term_map[normalized]:
                candidates[surface] = self.term_freq.get(normalized, 0) * 3.0

        # Strategy 2: Substring match
        for term_key, surface_forms in self.term_map.items():
            if term_key == normalized:
                continue
            if normalized in term_key or term_key in normalized:
                freq = self.term_freq.get(term_key, 0)
                for surface in surface_forms:
                    score = candidates.get(surface, 0)
                    candidates[surface] = max(score, freq * 2.0)

        # Strategy 3: Common prefix (≥3 chars)
        if len(normalized) >= 3:
            prefix = normalized[:3]
            for term_key, surface_forms in self.term_map.items():
                if term_key.startswith(prefix) and term_key != normalized:
                    freq = self.term_freq.get(term_key, 0)
                    for surface in surface_forms:
                        if surface not in candidates:
                            candidates[surface] = freq * 1.0

        # Rank by score (frequency-weighted) and return top_k
        ranked = sorted(candidates.items(), key=lambda x: x[1], reverse=True)
        return [term for term, _ in ranked[:top_k]]

    def get_top_terms(self, n: int = 50) -> list[tuple[str, int]]:
        """Return the most frequent terms in the corpus."""
        return self.term_freq.most_common(n)

    def save(self, path: Path) -> None:
        """Serialize the index to a JSON file."""
        data = {
            "term_map": {k: sorted(v) for k, v in self.term_map.items()},
            "term_freq": dict(self.term_freq.most_common()),
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def load(self, path: Path, passages: list[Passage] | None = None) -> None:
        """Load the index from a JSON file.

        If the file doesn't exist, optionally rebuild from passages.
        """
        if path.exists():
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            self.term_map = defaultdict(
                set, {k: set(v) for k, v in data.get("term_map", {}).items()}
            )
            self.term_freq = Counter(data.get("term_freq", {}))
            self._all_terms = set()
            for surface_forms in self.term_map.values():
                self._all_terms.update(surface_forms)
        elif passages:
            self.build(passages)

    @staticmethod
    def _extract_terms(text: str) -> list[str]:
        """Extract meaningful terms from text.

        Extracts:
        - CamelCase words (e.g., HttpRequest, DataHub)
        - Korean words (≥2 chars)
        - English words (≥3 chars)
        - Compound terms with hyphens/underscores
        """
        terms = []

        # CamelCase terms
        camel_pattern = re.compile(r"[A-Z][a-z]+(?:[A-Z][a-z]+)+")
        terms.extend(camel_pattern.findall(text))

        # Hyphenated/underscored compounds
        compound_pattern = re.compile(r"[a-zA-Z가-힣][\w-]*[-_][\w-]*[a-zA-Z가-힣]")
        terms.extend(compound_pattern.findall(text))

        # Korean words (≥2 chars)
        korean_pattern = re.compile(r"[가-힣]{2,}")
        terms.extend(korean_pattern.findall(text))

        # English words (≥3 chars, not stopwords)
        english_pattern = re.compile(r"\b[a-zA-Z]{3,}\b")
        stopwords = {
            "the",
            "and",
            "for",
            "are",
            "but",
            "not",
            "you",
            "all",
            "can",
            "had",
            "her",
            "was",
            "one",
            "our",
            "out",
            "has",
            "with",
            "that",
            "this",
            "from",
            "they",
            "been",
            "have",
            "will",
            "each",
            "make",
            "like",
            "than",
            "them",
            "then",
            "into",
            "some",
            "when",
            "what",
            "which",
            "their",
            "about",
        }
        for match in english_pattern.findall(text):
            if match.lower() not in stopwords:
                terms.append(match)

        return terms
