"""Tests for ReAct-based agentic retrieval refinement.

Covers:
  - SectionIndex build and search
  - TermIndex build and lookup
  - Tool functions (mocked retriever/indexer/evaluator)
  - Agent signature field validation
  - Settings and ablation flags
  - DocumentIndexer auxiliary index integration
  - Trajectory parsing utilities
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import dspy
import pytest

from agentic_rag.config.settings import settings
from agentic_rag.pipeline.agentic import parse_action_history, parse_evaluation_scores
from agentic_rag.retriever.indexer import DocumentIndexer, Passage
from agentic_rag.retriever.section_index import SectionIndex
from agentic_rag.retriever.term_index import TermIndex
from agentic_rag.signatures.agent import AgenticRefinementSignature


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def passages() -> list[Passage]:
    """Sample passages for index testing."""
    return [
        Passage(
            id="p1",
            title="Atelier REST API Guide",
            content="Atelier provides HttpRequest DSL function for external API calls.",
            source="docs/atelier.md",
        ),
        Passage(
            id="p2",
            title="Atelier REST API Guide",
            content="The HttpRequest function supports GET, POST, PUT methods.",
            source="docs/atelier.md",
        ),
        Passage(
            id="p3",
            title="Py-Runner Module Overview",
            content="Py-Runner executes Python scripts within Atelier workflows.",
            source="docs/py-runner.md",
        ),
        Passage(
            id="p4",
            title="DataHub 연동 가이드",
            content="DataHub는 데이터 파이프라인 관리를 위한 마이크로서비스입니다.",
            source="docs/datahub.md",
        ),
        Passage(
            id="p5",
            title="DSL 함수 레퍼런스",
            content="ExternalConnector 함수로 외부 시스템과 연동할 수 있습니다.",
            source="docs/dsl-reference.md",
        ),
    ]


# ---------------------------------------------------------------------------
# SectionIndex Tests
# ---------------------------------------------------------------------------
class TestSectionIndex:
    def test_build_and_search(self, passages):
        idx = SectionIndex()
        idx.build(passages)

        # Should have 4 unique sources
        sources = idx.get_sources()
        assert len(sources) == 4

        # Search by keyword
        results = idx.search("Atelier")
        assert len(results) >= 1
        assert any("Atelier" in r["title"] for r in results)

    def test_search_empty_keyword_returns_all(self, passages):
        idx = SectionIndex()
        idx.build(passages)

        results = idx.search("")
        # Should return all unique (source, title) combos
        assert len(results) >= 4

    def test_search_no_match(self, passages):
        idx = SectionIndex()
        idx.build(passages)

        results = idx.search("NonexistentTopic")
        assert results == []

    def test_get_sections_for_source(self, passages):
        idx = SectionIndex()
        idx.build(passages)

        sections = idx.get_sections_for_source("docs/atelier.md")
        assert len(sections) == 1  # deduplicated by title
        assert sections[0]["title"] == "Atelier REST API Guide"
        assert sections[0]["passage_count"] == 2  # p1 and p2

    def test_save_and_load(self, passages, tmp_dir):
        idx = SectionIndex()
        idx.build(passages)

        path = tmp_dir / "section_index.json"
        idx.save(path)

        idx2 = SectionIndex()
        idx2.load(path)
        assert idx2.get_sources() == idx.get_sources()

    def test_substring_matching(self, passages):
        idx = SectionIndex()
        idx.build(passages)

        # "API" should match "Atelier REST API Guide"
        results = idx.search("API")
        assert len(results) >= 1


# ---------------------------------------------------------------------------
# TermIndex Tests
# ---------------------------------------------------------------------------
class TestTermIndex:
    def test_build_and_lookup(self, passages):
        idx = TermIndex()
        idx.build(passages)

        # "API" should find related terms
        results = idx.lookup("API")
        assert len(results) > 0

    def test_lookup_korean_term(self, passages):
        idx = TermIndex()
        idx.build(passages)

        # "데이터" should find DataHub-related terms
        results = idx.lookup("데이터")
        assert len(results) > 0

    def test_lookup_no_match(self, passages):
        idx = TermIndex()
        idx.build(passages)

        results = idx.lookup("zzzznonexistent")
        assert results == []

    def test_lookup_empty(self, passages):
        idx = TermIndex()
        idx.build(passages)

        results = idx.lookup("")
        assert results == []

    def test_camelcase_extraction(self, passages):
        idx = TermIndex()
        idx.build(passages)

        # "HttpRequest" is a CamelCase term in the content
        results = idx.lookup("HttpRequest")
        assert any("HttpRequest" in r for r in results)

    def test_save_and_load(self, passages, tmp_dir):
        idx = TermIndex()
        idx.build(passages)

        path = tmp_dir / "term_index.json"
        idx.save(path)

        idx2 = TermIndex()
        idx2.load(path)
        assert idx2.lookup("API") == idx.lookup("API")

    def test_get_top_terms(self, passages):
        idx = TermIndex()
        idx.build(passages)

        top = idx.get_top_terms(10)
        assert len(top) > 0
        assert all(isinstance(t, tuple) and len(t) == 2 for t in top)


# ---------------------------------------------------------------------------
# Agent Signature Tests
# ---------------------------------------------------------------------------
class TestAgenticRefinementSignature:
    def test_input_fields(self):
        fields = AgenticRefinementSignature.input_fields
        expected = {
            "question",
            "initial_query",
            "initial_keywords",
            "quality_threshold",
            "max_passages",
        }
        assert set(fields.keys()) == expected

    def test_output_fields(self):
        fields = AgenticRefinementSignature.output_fields
        expected = {
            "final_passages",
            "final_action",
        }
        assert set(fields.keys()) == expected

    def test_react_module_creation(self):
        """Verify dspy.ReAct can be instantiated with our signature."""

        def dummy_tool(query: str) -> str:
            """A dummy tool for testing."""
            return "result"

        react = dspy.ReAct(AgenticRefinementSignature, tools=[dummy_tool])
        assert react is not None


# ---------------------------------------------------------------------------
# Trajectory Parsing Tests
# ---------------------------------------------------------------------------
class TestTrajectoryParsing:
    def test_parse_action_history_basic(self):
        trajectory = {
            "thought_0": "I should search first",
            "tool_name_0": "search_passages",
            "tool_args_0": {"query": "test"},
            "observation_0": '[{"id": "p1"}]',
            "thought_1": "Now evaluate",
            "tool_name_1": "evaluate_passages",
            "tool_args_1": {"question": "test", "passage_ids_json": '["p1"]'},
            "observation_1": '{"total": 70, "action": "output"}',
            "thought_2": "Quality is good, finish",
            "tool_name_2": "finish",
            "tool_args_2": {},
            "observation_2": "Completed.",
        }
        actions = parse_action_history(trajectory)
        assert actions == ["search_passages", "evaluate_passages"]

    def test_parse_action_history_empty(self):
        assert parse_action_history({}) == []

    def test_parse_evaluation_scores(self):
        trajectory = {
            "thought_0": "Search",
            "tool_name_0": "search_passages",
            "tool_args_0": {},
            "observation_0": "[]",
            "thought_1": "Evaluate",
            "tool_name_1": "evaluate_passages",
            "tool_args_1": {},
            "observation_1": json.dumps(
                {
                    "relevance": 20,
                    "coverage": 15,
                    "specificity": 18,
                    "sufficiency": 12,
                    "total": 65,
                    "action": "output",
                }
            ),
        }
        scores = parse_evaluation_scores(trajectory)
        assert len(scores) == 1
        assert scores[0]["total"] == 65

    def test_parse_evaluation_scores_multiple(self):
        trajectory = {
            "thought_0": "Evaluate first",
            "tool_name_0": "evaluate_passages",
            "tool_args_0": {},
            "observation_0": json.dumps({"total": 30, "action": "refine"}),
            "thought_1": "Search more",
            "tool_name_1": "search_passages",
            "tool_args_1": {},
            "observation_1": "[]",
            "thought_2": "Evaluate again",
            "tool_name_2": "evaluate_passages",
            "tool_args_2": {},
            "observation_2": json.dumps({"total": 70, "action": "output"}),
        }
        scores = parse_evaluation_scores(trajectory)
        assert len(scores) == 2
        assert scores[0]["total"] == 30
        assert scores[1]["total"] == 70

    def test_parse_evaluation_scores_malformed_json(self):
        trajectory = {
            "thought_0": "Evaluate",
            "tool_name_0": "evaluate_passages",
            "tool_args_0": {},
            "observation_0": "not valid json",
        }
        scores = parse_evaluation_scores(trajectory)
        assert scores == []


# ---------------------------------------------------------------------------
# Tool Tests (mocked)
# ---------------------------------------------------------------------------
class TestAgentTools:
    def test_search_passages_tool(self, passages):
        from agentic_rag.tools import create_tools

        mock_retriever = MagicMock()
        mock_retriever.search.return_value = [("p1", 0.95), ("p3", 0.80)]

        indexer = DocumentIndexer.__new__(DocumentIndexer)
        indexer.passages = passages
        indexer.section_index = SectionIndex()
        indexer.term_index = TermIndex()

        mock_evaluator = MagicMock()

        tools = create_tools(mock_retriever, indexer, mock_evaluator)
        search_fn = tools[0]  # search_passages

        result = json.loads(search_fn("test query", 5))
        assert isinstance(result, list)
        assert len(result) == 2
        assert result[0]["id"] == "p1"
        assert "content_preview" in result[0]

    def test_list_sections_tool(self, passages):
        from agentic_rag.tools import create_tools

        indexer = DocumentIndexer.__new__(DocumentIndexer)
        indexer.passages = passages
        indexer.section_index = SectionIndex()
        indexer.section_index.build(passages)
        indexer.term_index = TermIndex()

        tools = create_tools(MagicMock(), indexer, MagicMock())
        list_sections_fn = tools[1]

        result = json.loads(list_sections_fn("Atelier"))
        assert isinstance(result, list)
        assert len(result) >= 1

    def test_get_terminology_tool(self, passages):
        from agentic_rag.tools import create_tools

        indexer = DocumentIndexer.__new__(DocumentIndexer)
        indexer.passages = passages
        indexer.section_index = SectionIndex()
        indexer.term_index = TermIndex()
        indexer.term_index.build(passages)

        tools = create_tools(MagicMock(), indexer, MagicMock())
        get_term_fn = tools[2]

        result = json.loads(get_term_fn("API"))
        assert isinstance(result, list)

    def test_get_passage_detail_tool(self, passages):
        from agentic_rag.tools import create_tools

        indexer = DocumentIndexer.__new__(DocumentIndexer)
        indexer.passages = passages
        indexer.section_index = SectionIndex()
        indexer.term_index = TermIndex()

        tools = create_tools(MagicMock(), indexer, MagicMock())
        detail_fn = tools[4]

        result = json.loads(detail_fn("p1"))
        assert result["id"] == "p1"
        assert "content" in result

    def test_get_passage_detail_not_found(self, passages):
        from agentic_rag.tools import create_tools

        indexer = DocumentIndexer.__new__(DocumentIndexer)
        indexer.passages = passages
        indexer.section_index = SectionIndex()
        indexer.term_index = TermIndex()

        tools = create_tools(MagicMock(), indexer, MagicMock())
        detail_fn = tools[4]

        result = json.loads(detail_fn("nonexistent"))
        assert "error" in result

    def test_tool_error_handling(self):
        """Tools should return error JSON, not raise exceptions."""
        from agentic_rag.tools import create_tools

        mock_retriever = MagicMock()
        mock_retriever.search.side_effect = RuntimeError("Search failed")

        indexer = DocumentIndexer.__new__(DocumentIndexer)
        indexer.passages = []
        indexer.section_index = SectionIndex()
        indexer.term_index = TermIndex()

        tools = create_tools(mock_retriever, indexer, MagicMock())
        search_fn = tools[0]

        result = json.loads(search_fn("test"))
        assert "error" in result


# ---------------------------------------------------------------------------
# Settings Tests
# ---------------------------------------------------------------------------
class TestAgentSettings:
    def test_agent_settings_defaults(self):
        assert settings.agent.max_iterations == 15

    def test_enable_agentic_refinement_default_false(self):
        assert settings.experiment.enable_agentic_refinement is False

    def test_ablation_flags_count(self):
        """Total of 7 ablation flags: C1-C5 (6 flags) + agentic refinement."""
        exp = settings.experiment
        flags = [
            exp.enable_iteration,
            exp.enable_accumulation,
            exp.enable_4d_evaluation,
            exp.enable_refinement,
            exp.enable_agent_routing,
            exp.enable_dspy,
            exp.enable_agentic_refinement,
        ]
        assert len(flags) == 7
        # C1-C5 default True, agentic refinement default False
        assert sum(flags) == 6


# ---------------------------------------------------------------------------
# DocumentIndexer Integration Tests
# ---------------------------------------------------------------------------
class TestIndexerIntegration:
    def test_indexer_has_auxiliary_indices(self):
        indexer = DocumentIndexer()
        assert hasattr(indexer, "section_index")
        assert hasattr(indexer, "term_index")
        assert isinstance(indexer.section_index, SectionIndex)
        assert isinstance(indexer.term_index, TermIndex)
