"""Document indexer — loads raw documents and builds retrieval indices.

Handles document loading, chunking, and index construction for both
dense (FAISS) and sparse (BM25) retrievers.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from loguru import logger

from agentic_rag.config.settings import settings
from agentic_rag.retriever.hybrid import HybridRetriever
from agentic_rag.retriever.section_index import SectionIndex
from agentic_rag.retriever.term_index import TermIndex


@dataclass
class Passage:
    """A single retrievable passage/chunk."""

    id: str
    title: str
    content: str
    source: str = ""
    metadata: dict = field(default_factory=dict)


class DocumentIndexer:
    """Load documents, chunk them, and build retrieval indices."""

    def __init__(self, retriever: HybridRetriever | None = None):
        self.retriever = retriever or HybridRetriever()
        self.passages: list[Passage] = []
        self.section_index = SectionIndex()
        self.term_index = TermIndex()

    # ------------------------------------------------------------------
    # Document loading
    # ------------------------------------------------------------------
    def load_jsonl(self, path: Path) -> list[Passage]:
        """Load passages from a JSONL file.

        Expected format per line:
        {"id": "...", "title": "...", "content": "...", "source": "...", ...}
        """
        passages = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                data = json.loads(line.strip())
                p = Passage(
                    id=data.get("id", str(uuid.uuid4())),
                    title=data.get("title", ""),
                    content=data["content"],
                    source=data.get("source", str(path)),
                    metadata={
                        k: v
                        for k, v in data.items()
                        if k not in {"id", "title", "content", "source"}
                    },
                )
                passages.append(p)
        logger.info(f"Loaded {len(passages)} passages from {path}")
        return passages

    def load_json(self, path: Path) -> list[Passage]:
        """Load passages from a JSON array file."""
        with open(path, encoding="utf-8") as f:
            items = json.load(f)
        passages = [
            Passage(
                id=item.get("id", str(uuid.uuid4())),
                title=item.get("title", ""),
                content=item["content"],
                source=item.get("source", str(path)),
                metadata={
                    k: v for k, v in item.items() if k not in {"id", "title", "content", "source"}
                },
            )
            for item in items
        ]
        logger.info(f"Loaded {len(passages)} passages from {path}")
        return passages

    # ------------------------------------------------------------------
    # Chunking
    # ------------------------------------------------------------------
    @staticmethod
    def chunk_text(
        text: str,
        chunk_size: int = 512,
        overlap: int = 64,
    ) -> list[str]:
        """Split text into overlapping word-level chunks."""
        words = text.split()
        if len(words) <= chunk_size:
            return [text]

        chunks = []
        start = 0
        while start < len(words):
            end = start + chunk_size
            chunk = " ".join(words[start:end])
            chunks.append(chunk)
            start = end - overlap
        return chunks

    def chunk_passages(
        self,
        passages: list[Passage],
        chunk_size: int = 512,
        overlap: int = 64,
    ) -> list[Passage]:
        """Split large passages into smaller chunks."""
        chunked: list[Passage] = []
        for p in passages:
            chunks = self.chunk_text(p.content, chunk_size, overlap)
            if len(chunks) == 1:
                chunked.append(p)
            else:
                for i, chunk in enumerate(chunks):
                    chunked.append(
                        Passage(
                            id=f"{p.id}_chunk{i}",
                            title=p.title,
                            content=chunk,
                            source=p.source,
                            metadata={**p.metadata, "chunk_index": i},
                        )
                    )
        logger.info(f"Chunked {len(passages)} passages → {len(chunked)} chunks")
        return chunked

    # ------------------------------------------------------------------
    # Index building
    # ------------------------------------------------------------------
    def build(
        self,
        data_paths: list[Path],
        chunk_size: int = 512,
        overlap: int = 64,
    ) -> HybridRetriever:
        """Load documents, chunk, and build the hybrid index."""
        all_passages: list[Passage] = []
        for path in data_paths:
            if path.suffix == ".jsonl":
                all_passages.extend(self.load_jsonl(path))
            elif path.suffix == ".json":
                all_passages.extend(self.load_json(path))
            else:
                logger.warning(f"Unsupported format: {path}")

        self.passages = self.chunk_passages(all_passages, chunk_size, overlap)
        self.retriever.build_index(self.passages)

        # Build auxiliary indices for RLM tools
        self.section_index.build(self.passages)
        self.term_index.build(self.passages)
        logger.info(
            f"Built section index ({len(self.section_index.source_to_sections)} sources) "
            f"and term index ({len(self.term_index.term_map)} terms)"
        )

        return self.retriever

    def save(self, index_dir: Path | None = None) -> None:
        """Save indices and passage data to disk."""
        index_dir = index_dir or settings.index_dir
        self.retriever.save(index_dir)

        # Also save passage metadata for later lookup
        passage_data = [
            {
                "id": p.id,
                "title": p.title,
                "content": p.content,
                "source": p.source,
                **p.metadata,
            }
            for p in self.passages
        ]
        with open(index_dir / "passages.jsonl", "w", encoding="utf-8") as f:
            for item in passage_data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

        # Save auxiliary indices
        self.section_index.save(index_dir / "section_index.json")
        self.term_index.save(index_dir / "term_index.json")
        logger.info(f"Index saved to {index_dir}")

    def load(self, index_dir: Path | None = None) -> HybridRetriever:
        """Load indices and passage data from disk."""
        index_dir = index_dir or settings.index_dir
        self.retriever.load(index_dir)

        self.passages = []
        passages_file = index_dir / "passages.jsonl"
        if passages_file.exists():
            self.passages = self.load_jsonl(passages_file)

        # Load auxiliary indices
        self.section_index.load(index_dir / "section_index.json", self.passages)
        self.term_index.load(index_dir / "term_index.json", self.passages)

        return self.retriever

    # ------------------------------------------------------------------
    # Passage lookup
    # ------------------------------------------------------------------
    def get_passage(self, passage_id: str) -> Passage | None:
        """Look up a passage by ID."""
        for p in self.passages:
            if p.id == passage_id:
                return p
        return None

    def get_passages(self, passage_ids: list[str]) -> list[Passage]:
        """Look up multiple passages by ID, preserving order."""
        id_map = {p.id: p for p in self.passages}
        return [id_map[pid] for pid in passage_ids if pid in id_map]
