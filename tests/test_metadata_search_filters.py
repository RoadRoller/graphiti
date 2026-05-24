"""
Tests for Phase 3: metadata filtering in SearchFilters and query constructors.
"""
from __future__ import annotations

import pytest

from graphiti_core.driver.driver import GraphProvider
from graphiti_core.search.search_filters import (
    SearchFilters,
    edge_search_filter_query_constructor,
    node_search_filter_query_constructor,
)

pytest_plugins = ('pytest_asyncio',)


# ---------------------------------------------------------------------------
# SearchFilters model validation
# ---------------------------------------------------------------------------


def test_search_filters_accepts_metadata():
    filters = SearchFilters(metadata={'agent_id': 42, 'session': 'abc'})
    assert filters.metadata == {'agent_id': 42, 'session': 'abc'}


def test_search_filters_metadata_defaults_to_none():
    filters = SearchFilters()
    assert filters.metadata is None


def test_search_filters_accepts_empty_metadata():
    filters = SearchFilters(metadata={})
    assert filters.metadata == {}


def test_search_filters_rejects_nested_metadata_value():
    """Metadata values must be str | int | float — nested dicts should be rejected."""
    with pytest.raises(Exception):
        SearchFilters(metadata={'key': {'nested': 'value'}})  # type: ignore[arg-type]


def test_property_filters_field_removed():
    """property_filters must no longer exist on SearchFilters."""
    assert not hasattr(SearchFilters(), 'property_filters')


# ---------------------------------------------------------------------------
# node_search_filter_query_constructor – metadata
# ---------------------------------------------------------------------------


def test_node_filter_single_metadata_neo4j():
    filters = SearchFilters(metadata={'agent_id': 42})
    queries, params = node_search_filter_query_constructor(filters, GraphProvider.NEO4J)

    assert queries == ['n.metadata_agent_id = $node_metadata_value_0']
    assert params == {'node_metadata_value_0': 42}


def test_node_filter_multiple_metadata_neo4j():
    filters = SearchFilters(metadata={'agent_id': 42, 'research_subject_id': 123})
    queries, params = node_search_filter_query_constructor(filters, GraphProvider.NEO4J)

    assert len(queries) == 2
    assert 'n.metadata_agent_id = $node_metadata_value_0' in queries
    assert 'n.metadata_research_subject_id = $node_metadata_value_1' in queries
    assert params['node_metadata_value_0'] == 42
    assert params['node_metadata_value_1'] == 123


def test_node_filter_metadata_string_value_neo4j():
    filters = SearchFilters(metadata={'session_id': 'abc123'})
    queries, params = node_search_filter_query_constructor(filters, GraphProvider.NEO4J)

    assert queries == ['n.metadata_session_id = $node_metadata_value_0']
    assert params == {'node_metadata_value_0': 'abc123'}


def test_node_filter_metadata_float_value_neo4j():
    filters = SearchFilters(metadata={'score': 0.95})
    queries, params = node_search_filter_query_constructor(filters, GraphProvider.NEO4J)

    assert queries == ['n.metadata_score = $node_metadata_value_0']
    assert params == {'node_metadata_value_0': 0.95}


def test_node_filter_single_metadata_kuzu():
    filters = SearchFilters(metadata={'agent_id': 42})
    queries, params = node_search_filter_query_constructor(filters, GraphProvider.KUZU)

    assert len(queries) == 1
    assert "json_extract(n.metadata, '$.agent_id') = $node_metadata_value_0" in queries
    assert params == {'node_metadata_value_0': 42}


def test_node_filter_metadata_combined_with_node_labels_neo4j():
    filters = SearchFilters(node_labels=['Person'], metadata={'agent_id': 42})
    queries, params = node_search_filter_query_constructor(filters, GraphProvider.NEO4J)

    assert 'n:Person' in queries
    assert 'n.metadata_agent_id = $node_metadata_value_0' in queries
    assert len(queries) == 2
    assert params['node_metadata_value_0'] == 42


def test_node_filter_no_metadata_no_extra_queries():
    filters = SearchFilters(node_labels=['Person'])
    queries, params = node_search_filter_query_constructor(filters, GraphProvider.NEO4J)

    assert queries == ['n:Person']
    assert not any('metadata' in q for q in queries)


def test_node_filter_empty_metadata_no_extra_queries():
    filters = SearchFilters(metadata={})
    queries, params = node_search_filter_query_constructor(filters, GraphProvider.NEO4J)

    # Empty metadata dict: no filters should be added
    assert queries == []
    assert params == {}


# ---------------------------------------------------------------------------
# edge_search_filter_query_constructor – metadata
# ---------------------------------------------------------------------------


def test_edge_filter_single_metadata_neo4j():
    filters = SearchFilters(metadata={'agent_id': 42})
    queries, params = edge_search_filter_query_constructor(filters, GraphProvider.NEO4J)

    assert queries == ['e.metadata_agent_id = $edge_metadata_value_0']
    assert params == {'edge_metadata_value_0': 42}


def test_edge_filter_multiple_metadata_neo4j():
    filters = SearchFilters(metadata={'agent_id': 42, 'session_id': 'xyz'})
    queries, params = edge_search_filter_query_constructor(filters, GraphProvider.NEO4J)

    assert len(queries) == 2
    assert 'e.metadata_agent_id = $edge_metadata_value_0' in queries
    assert 'e.metadata_session_id = $edge_metadata_value_1' in queries
    assert params['edge_metadata_value_0'] == 42
    assert params['edge_metadata_value_1'] == 'xyz'


def test_edge_filter_metadata_kuzu():
    filters = SearchFilters(metadata={'agent_id': 7})
    queries, params = edge_search_filter_query_constructor(filters, GraphProvider.KUZU)

    assert len(queries) == 1
    assert "json_extract(e.metadata, '$.agent_id') = $edge_metadata_value_0" in queries
    assert params == {'edge_metadata_value_0': 7}


def test_edge_filter_metadata_combined_with_edge_types():
    filters = SearchFilters(edge_types=['WORKS_AT'], metadata={'agent_id': 1})
    queries, params = edge_search_filter_query_constructor(filters, GraphProvider.NEO4J)

    assert 'e.name in $edge_types' in queries
    assert 'e.metadata_agent_id = $edge_metadata_value_0' in queries
    assert params['edge_types'] == ['WORKS_AT']
    assert params['edge_metadata_value_0'] == 1


def test_edge_filter_no_metadata_no_extra_queries():
    filters = SearchFilters(edge_types=['KNOWS'])
    queries, params = edge_search_filter_query_constructor(filters, GraphProvider.NEO4J)

    assert not any('metadata' in q for q in queries)


def test_edge_filter_empty_metadata_no_extra_queries():
    filters = SearchFilters(metadata={})
    queries, params = edge_search_filter_query_constructor(filters, GraphProvider.NEO4J)

    assert queries == []
    assert params == {}


# ---------------------------------------------------------------------------
# Integration: node and edge filter param namespacing (no key collisions)
# ---------------------------------------------------------------------------


def test_node_and_edge_metadata_params_do_not_collide():
    """node and edge constructors use distinct param name prefixes."""
    filters = SearchFilters(metadata={'agent_id': 99})

    node_queries, node_params = node_search_filter_query_constructor(filters, GraphProvider.NEO4J)
    edge_queries, edge_params = edge_search_filter_query_constructor(filters, GraphProvider.NEO4J)

    # Make sure the param keys are different
    assert set(node_params.keys()).isdisjoint(set(edge_params.keys()))
    assert 'node_metadata_value_0' in node_params
    assert 'edge_metadata_value_0' in edge_params
