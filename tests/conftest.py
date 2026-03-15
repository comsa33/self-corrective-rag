"""Shared test fixtures for all test modules."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def _set_test_env(monkeypatch):
    """Set test environment variables."""
    monkeypatch.setenv("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY", "test-key-for-unit-tests"))


@pytest.fixture
def tmp_dir():
    """Provide a temporary directory for test artifacts."""
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)


@pytest.fixture
def sample_passages() -> list[dict]:
    """Sample passages for retriever/pipeline testing."""
    return [
        {
            "id": "doc_1",
            "title": "Python Basics",
            "content": "Python is a high-level programming language known for its readability.",
        },
        {
            "id": "doc_2",
            "title": "Machine Learning",
            "content": "Machine learning is a subset of AI that learns from data.",
        },
        {
            "id": "doc_3",
            "title": "RAG Systems",
            "content": "Retrieval-Augmented Generation combines retrieval with LLM generation.",
        },
        {
            "id": "doc_4",
            "title": "DSPy Framework",
            "content": "DSPy is a framework for programming foundation models declaratively.",
        },
        {
            "id": "doc_5",
            "title": "Vector Search",
            "content": "FAISS is a library for efficient similarity search and clustering.",
        },
    ]


@pytest.fixture
def sample_qa_pairs() -> list[dict]:
    """Sample Q&A pairs for pipeline testing."""
    return [
        {
            "id": "qa_1",
            "question": "What is Python?",
            "answer": "a high-level programming language",
        },
        {
            "id": "qa_2",
            "question": "What is RAG?",
            "answer": "Retrieval-Augmented Generation",
        },
        {
            "id": "qa_3",
            "question": "What is DSPy?",
            "answer": "a framework for programming foundation models",
        },
    ]
