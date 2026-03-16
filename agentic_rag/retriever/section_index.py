"""Section-level index for document structure browsing.

Provides a lightweight index over document sections/titles, enabling
agent tools to browse document structure before committing to full-text search.
No LLM calls — pure in-memory data structures built from passage metadata.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agentic_rag.retriever.indexer import Passage


@dataclass
class SectionEntry:
    """A document section with its source and associated passage IDs."""

    title: str
    source: str
    passage_ids: list[str] = field(default_factory=list)


class SectionIndex:
    """Index mapping document sources to their section titles.

    Built from passage titles and sources during indexing.
    Enables agents to browse document structure and identify
    promising sections before performing full-text retrieval.
    """

    def __init__(self) -> None:
        # source_doc → [SectionEntry]
        self.source_to_sections: dict[str, list[SectionEntry]] = defaultdict(list)
        # normalized keyword → [(source, title, passage_ids)]
        self._keyword_index: dict[str, list[SectionEntry]] = defaultdict(list)

    def build(self, passages: list[Passage]) -> None:
        """Build the section index from passages.

        Groups passages by (source, title) to create section entries,
        then builds a keyword reverse index from title words.
        """
        self.source_to_sections.clear()
        self._keyword_index.clear()

        # Group by (source, title) to deduplicate chunks from the same section
        section_map: dict[tuple[str, str], list[str]] = defaultdict(list)
        for p in passages:
            key = (p.source, p.title)
            section_map[key].append(p.id)

        for (source, title), pids in section_map.items():
            entry = SectionEntry(title=title, source=source, passage_ids=pids)
            self.source_to_sections[source].append(entry)

            # Build keyword index from title words
            for word in self._tokenize(title):
                self._keyword_index[word].append(entry)

    def search(self, keyword: str = "") -> list[dict]:
        """Search for sections matching a keyword.

        Args:
            keyword: Search term to match against section titles.
                     Empty string returns all sections grouped by source.

        Returns:
            List of dicts: [{"source": ..., "title": ..., "passage_count": ...}]
        """
        if not keyword.strip():
            return self._all_sections()

        results: list[dict] = []
        seen: set[tuple[str, str]] = set()

        for token in self._tokenize(keyword):
            for entry in self._keyword_index.get(token, []):
                key = (entry.source, entry.title)
                if key not in seen:
                    seen.add(key)
                    results.append(
                        {
                            "source": entry.source,
                            "title": entry.title,
                            "passage_count": len(entry.passage_ids),
                            "passage_ids": entry.passage_ids[:5],  # preview
                        }
                    )

        # Also do substring matching on full titles
        kw_lower = keyword.lower()
        for _source, sections in self.source_to_sections.items():
            for entry in sections:
                key = (entry.source, entry.title)
                if key not in seen and kw_lower in entry.title.lower():
                    seen.add(key)
                    results.append(
                        {
                            "source": entry.source,
                            "title": entry.title,
                            "passage_count": len(entry.passage_ids),
                            "passage_ids": entry.passage_ids[:5],
                        }
                    )

        return results

    def get_sources(self) -> list[str]:
        """Return all indexed document sources."""
        return list(self.source_to_sections.keys())

    def get_sections_for_source(self, source: str) -> list[dict]:
        """Return all sections for a specific document source."""
        return [
            {
                "title": entry.title,
                "passage_count": len(entry.passage_ids),
                "passage_ids": entry.passage_ids[:5],
            }
            for entry in self.source_to_sections.get(source, [])
        ]

    def _all_sections(self) -> list[dict]:
        """Return all sections across all sources."""
        results = []
        for source, sections in self.source_to_sections.items():
            for entry in sections:
                results.append(
                    {
                        "source": source,
                        "title": entry.title,
                        "passage_count": len(entry.passage_ids),
                    }
                )
        return results

    def save(self, path: Path) -> None:
        """Serialize the index to a JSON file."""
        data = {}
        for source, sections in self.source_to_sections.items():
            data[source] = [{"title": s.title, "passage_ids": s.passage_ids} for s in sections]
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def load(self, path: Path, passages: list[Passage] | None = None) -> None:
        """Load the index from a JSON file.

        If the file doesn't exist, optionally rebuild from passages.
        """
        if path.exists():
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            self.source_to_sections.clear()
            self._keyword_index.clear()
            for source, sections in data.items():
                for s in sections:
                    entry = SectionEntry(
                        title=s["title"],
                        source=source,
                        passage_ids=s.get("passage_ids", []),
                    )
                    self.source_to_sections[source].append(entry)
                    for word in self._tokenize(entry.title):
                        self._keyword_index[word].append(entry)
        elif passages:
            self.build(passages)

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """Simple lowercase word tokenization."""
        return [w.lower().strip() for w in text.split() if w.strip()]
