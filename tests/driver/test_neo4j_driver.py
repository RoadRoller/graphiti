"""
Phase 4.1 – Neo4j-specific metadata tests.

Verifies:
- EntityNode metadata is stored with the ``metadata_`` prefix as individual Neo4j
  node properties (not as a nested map or a single ``metadata`` property).
- EntityEdge metadata is stored with the ``metadata_`` prefix as individual Neo4j
  relationship properties.
- After a round-trip, metadata keys do not leak into the ``attributes`` dict and
  the ``attributes`` dict values survive unchanged.
- The node/edge search-filter query constructors emit the correct prefixed Cypher
  for Neo4j and use distinct parameter-name namespaces so that node and edge
  params never collide.
"""

from __future__ import annotations

import numpy as np
import os
import pytest
from datetime import datetime

from graphiti_core.driver.driver import GraphProvider
from graphiti_core.edges import EntityEdge
from graphiti_core.nodes import EntityNode
from graphiti_core.search.search_filters import (
    SearchFilters,
    edge_search_filter_query_constructor,
    node_search_filter_query_constructor,
)

try:
    from graphiti_core.driver.neo4j_driver import Neo4jDriver

    HAS_NEO4J = True
except ImportError:  # pragma: no cover
    HAS_NEO4J = False

pytest_plugins = ('pytest_asyncio',)

NEO4J_URI = os.getenv('NEO4J_URI', 'bolt://localhost:7687')
NEO4J_USER = os.getenv('NEO4J_USER', 'neo4j')
NEO4J_PASSWORD = os.getenv('NEO4J_PASSWORD', 'test')

_GROUP_ID = 'neo4j_driver_metadata_test_group'


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
async def neo4j_driver():
    if not HAS_NEO4J:
        pytest.skip('Neo4j driver is not installed')
    if os.getenv('DISABLE_NEO4J'):
        pytest.skip('Neo4j disabled via DISABLE_NEO4J env var')
    driver = Neo4jDriver(uri=NEO4J_URI, user=NEO4J_USER, password=NEO4J_PASSWORD)
    try:
        yield driver
    finally:
        await driver.execute_query(
            'MATCH (n {group_id: $gid}) DETACH DELETE n',
            gid=_GROUP_ID,
        )
        await driver.close()


def _rand_embedding() -> list[float]:
    return np.random.default_rng().uniform(0, 1, 384).tolist()


# ---------------------------------------------------------------------------
# EntityNode – raw Neo4j property verification
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_entity_node_metadata_stored_with_prefix(neo4j_driver):
    """
    Saving an EntityNode with metadata must produce ``metadata_<key>`` properties
    directly on the Neo4j node – not a nested ``metadata`` map property.
    """
    metadata = {'agent_id': 42, 'research_subject_id': 123}
    node = EntityNode(
        name='PrefixTestNode',
        group_id=_GROUP_ID,
        labels=['Entity'],
        created_at=datetime.now(),
        summary='',
        metadata=metadata,
    )
    node.name_embedding = _rand_embedding()
    await node.save(neo4j_driver)

    records, _, _ = await neo4j_driver.execute_query(
        'MATCH (n:Entity {uuid: $uuid}) RETURN properties(n) AS props',
        uuid=node.uuid,
        routing_='r',
    )
    assert len(records) == 1
    props: dict = records[0]['props']

    assert 'metadata_agent_id' in props, (
        f'Expected metadata_agent_id in node properties, got: {sorted(props)}'
    )
    assert 'metadata_research_subject_id' in props
    assert props['metadata_agent_id'] == 42
    assert props['metadata_research_subject_id'] == 123
    # The raw key ``metadata`` must NOT be stored (no nested-map property)
    assert 'metadata' not in props


@pytest.mark.asyncio
async def test_entity_node_no_metadata_no_prefix_properties(neo4j_driver):
    """
    Saving an EntityNode without metadata must not produce any ``metadata_*``
    properties on the Neo4j node.
    """
    node = EntityNode(
        name='NoMetaNode',
        group_id=_GROUP_ID,
        labels=['Entity'],
        created_at=datetime.now(),
        summary='',
    )
    node.name_embedding = _rand_embedding()
    await node.save(neo4j_driver)

    records, _, _ = await neo4j_driver.execute_query(
        'MATCH (n:Entity {uuid: $uuid}) RETURN properties(n) AS props',
        uuid=node.uuid,
        routing_='r',
    )
    assert len(records) == 1
    props: dict = records[0]['props']

    metadata_props = [k for k in props if k.startswith('metadata_')]
    assert metadata_props == [], f'Expected no metadata_ properties but found: {metadata_props}'


@pytest.mark.asyncio
async def test_entity_node_metadata_does_not_leak_into_attributes(neo4j_driver):
    """
    After a round-trip through Neo4j, metadata keys must appear in
    ``node.metadata`` and must NOT appear in ``node.attributes``.
    """
    metadata = {'user_flag': 'premium'}
    attributes = {'age': 25, 'city': 'Berlin'}
    node = EntityNode(
        name='IsolationNode',
        group_id=_GROUP_ID,
        labels=['Entity'],
        created_at=datetime.now(),
        summary='',
        attributes=attributes,
        metadata=metadata,
    )
    node.name_embedding = _rand_embedding()
    await node.save(neo4j_driver)

    retrieved = await EntityNode.get_by_uuid(neo4j_driver, node.uuid)

    assert retrieved.metadata == metadata
    assert 'user_flag' not in retrieved.attributes
    assert retrieved.attributes.get('age') == 25
    assert retrieved.attributes.get('city') == 'Berlin'


# ---------------------------------------------------------------------------
# EntityEdge – raw Neo4j property verification
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_entity_edge_metadata_stored_with_prefix(neo4j_driver):
    """
    Saving an EntityEdge with metadata must produce ``metadata_<key>`` properties
    directly on the Neo4j relationship – not a nested ``metadata`` map property.
    """
    metadata = {'session_id': 'abc', 'agent_id': 7}
    now = datetime.now()

    source = EntityNode(
        name='SrcPrefixNode', group_id=_GROUP_ID, labels=['Entity'], created_at=now, summary=''
    )
    target = EntityNode(
        name='TgtPrefixNode', group_id=_GROUP_ID, labels=['Entity'], created_at=now, summary=''
    )
    source.name_embedding = _rand_embedding()
    target.name_embedding = _rand_embedding()
    await source.save(neo4j_driver)
    await target.save(neo4j_driver)

    edge = EntityEdge(
        source_node_uuid=source.uuid,
        target_node_uuid=target.uuid,
        name='RELATES_TO',
        fact='SrcPrefixNode relates to TgtPrefixNode',
        group_id=_GROUP_ID,
        created_at=now,
        episodes=[],
        metadata=metadata,
    )
    edge.fact_embedding = _rand_embedding()
    await edge.save(neo4j_driver)

    records, _, _ = await neo4j_driver.execute_query(
        'MATCH ()-[e:RELATES_TO {uuid: $uuid}]->() RETURN properties(e) AS props',
        uuid=edge.uuid,
        routing_='r',
    )
    assert len(records) == 1
    props: dict = records[0]['props']

    assert 'metadata_session_id' in props, (
        f'Expected metadata_session_id in edge properties, got: {sorted(props)}'
    )
    assert 'metadata_agent_id' in props
    assert props['metadata_session_id'] == 'abc'
    assert props['metadata_agent_id'] == 7
    # The raw key ``metadata`` must NOT be stored
    assert 'metadata' not in props


@pytest.mark.asyncio
async def test_entity_edge_no_metadata_no_prefix_properties(neo4j_driver):
    """
    Saving an EntityEdge without metadata must not produce any ``metadata_*``
    properties on the Neo4j relationship.
    """
    now = datetime.now()

    source = EntityNode(
        name='SrcNoMeta', group_id=_GROUP_ID, labels=['Entity'], created_at=now, summary=''
    )
    target = EntityNode(
        name='TgtNoMeta', group_id=_GROUP_ID, labels=['Entity'], created_at=now, summary=''
    )
    source.name_embedding = _rand_embedding()
    target.name_embedding = _rand_embedding()
    await source.save(neo4j_driver)
    await target.save(neo4j_driver)

    edge = EntityEdge(
        source_node_uuid=source.uuid,
        target_node_uuid=target.uuid,
        name='RELATES_TO',
        fact='SrcNoMeta relates to TgtNoMeta',
        group_id=_GROUP_ID,
        created_at=now,
        episodes=[],
    )
    edge.fact_embedding = _rand_embedding()
    await edge.save(neo4j_driver)

    records, _, _ = await neo4j_driver.execute_query(
        'MATCH ()-[e:RELATES_TO {uuid: $uuid}]->() RETURN properties(e) AS props',
        uuid=edge.uuid,
        routing_='r',
    )
    assert len(records) == 1
    props: dict = records[0]['props']

    metadata_props = [k for k in props if k.startswith('metadata_')]
    assert metadata_props == [], (
        f'Expected no metadata_ properties on edge but found: {metadata_props}'
    )


@pytest.mark.asyncio
async def test_entity_edge_metadata_does_not_leak_into_attributes(neo4j_driver):
    """
    After a round-trip through Neo4j, edge metadata keys must appear in
    ``edge.metadata`` and must NOT appear in ``edge.attributes``.
    """
    metadata = {'run_id': 'xyz'}
    attributes = {'confidence': 0.9}
    now = datetime.now()

    source = EntityNode(
        name='SrcIsoEdge', group_id=_GROUP_ID, labels=['Entity'], created_at=now, summary=''
    )
    target = EntityNode(
        name='TgtIsoEdge', group_id=_GROUP_ID, labels=['Entity'], created_at=now, summary=''
    )
    source.name_embedding = _rand_embedding()
    target.name_embedding = _rand_embedding()
    await source.save(neo4j_driver)
    await target.save(neo4j_driver)

    edge = EntityEdge(
        source_node_uuid=source.uuid,
        target_node_uuid=target.uuid,
        name='RELATES_TO',
        fact='SrcIsoEdge relates to TgtIsoEdge',
        group_id=_GROUP_ID,
        created_at=now,
        episodes=[],
        attributes=attributes,
        metadata=metadata,
    )
    edge.fact_embedding = _rand_embedding()
    await edge.save(neo4j_driver)

    retrieved = await EntityEdge.get_by_uuid(neo4j_driver, edge.uuid)

    assert retrieved.metadata == metadata
    assert 'run_id' not in retrieved.attributes
    assert retrieved.attributes.get('confidence') == pytest.approx(0.9)


# ---------------------------------------------------------------------------
# Neo4j filter query construction (unit – no DB required)
# ---------------------------------------------------------------------------


def test_node_metadata_filter_generates_prefixed_cypher():
    """node_search_filter_query_constructor emits ``n.metadata_<key>`` for Neo4j."""
    filters = SearchFilters(metadata={'agent_id': 42, 'session': 'abc'})
    queries, params = node_search_filter_query_constructor(filters, GraphProvider.NEO4J)

    assert any('n.metadata_agent_id = $node_metadata_value_0' in q for q in queries)
    assert any('n.metadata_session = $node_metadata_value_1' in q for q in queries)
    assert params['node_metadata_value_0'] == 42
    assert params['node_metadata_value_1'] == 'abc'


def test_edge_metadata_filter_generates_prefixed_cypher():
    """edge_search_filter_query_constructor emits ``e.metadata_<key>`` for Neo4j."""
    filters = SearchFilters(metadata={'run_id': 'xyz', 'score': 0.5})
    queries, params = edge_search_filter_query_constructor(filters, GraphProvider.NEO4J)

    assert any('e.metadata_run_id = $edge_metadata_value_0' in q for q in queries)
    assert any('e.metadata_score = $edge_metadata_value_1' in q for q in queries)
    assert params['edge_metadata_value_0'] == 'xyz'
    assert params['edge_metadata_value_1'] == 0.5


def test_node_and_edge_metadata_param_names_are_disjoint():
    """
    Node and edge filter constructors must use different parameter-name prefixes
    so their params can be safely merged without key collisions.
    """
    filters = SearchFilters(metadata={'agent_id': 99})
    _, node_params = node_search_filter_query_constructor(filters, GraphProvider.NEO4J)
    _, edge_params = edge_search_filter_query_constructor(filters, GraphProvider.NEO4J)

    assert set(node_params).isdisjoint(set(edge_params)), (
        f'Param name collision between node ({set(node_params)}) '
        f'and edge ({set(edge_params)}) constructors'
    )
    assert 'node_metadata_value_0' in node_params
    assert 'edge_metadata_value_0' in edge_params


def test_node_metadata_filter_empty_dict_adds_no_queries():
    """Empty metadata dict must not add any filter clauses."""
    filters = SearchFilters(metadata={})
    queries, params = node_search_filter_query_constructor(filters, GraphProvider.NEO4J)
    assert queries == []
    assert params == {}


def test_edge_metadata_filter_empty_dict_adds_no_queries():
    """Empty metadata dict must not add any edge filter clauses."""
    filters = SearchFilters(metadata={})
    queries, params = edge_search_filter_query_constructor(filters, GraphProvider.NEO4J)
    assert queries == []
    assert params == {}
