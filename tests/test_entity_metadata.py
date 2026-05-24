"""
Tests for Phase 2: metadata field on EntityNode and EntityEdge.
Covers storage, retrieval, and propagation from episode metadata.
"""
from __future__ import annotations

import numpy as np
import pytest
from datetime import datetime
from unittest.mock import Mock

from graphiti_core.edges import (
    EntityEdge,
    EpisodicEdge,
)
from graphiti_core.nodes import (
    EntityNode,
    EpisodeType,
    EpisodicNode,
)
from graphiti_core.utils.bulk_utils import add_nodes_and_edges_bulk
from tests.helpers_test import (
    GraphProvider,
    group_id,
)

pytest_plugins = ('pytest_asyncio',)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_embedder():
    mock = Mock()
    rng = np.random.default_rng(0)

    async def _async_create(input_data):
        return rng.uniform(0, 1, 384).tolist()

    mock.create = _async_create
    return mock


# ---------------------------------------------------------------------------
# EntityNode metadata – save and retrieve
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_entity_node_metadata_save_and_get(graph_driver):
    """EntityNode.save() stores metadata; get_by_uuid() retrieves it."""
    now = datetime.now()
    metadata = {'agent_id': 42, 'research_subject_id': 123}

    node = EntityNode(
        name='Alice',
        group_id=group_id,
        labels=['Entity', 'Person'],
        created_at=now,
        summary='',
        metadata=metadata,
    )
    node.name_embedding = np.random.uniform(0, 1, 384).tolist()
    await node.save(graph_driver)

    retrieved = await EntityNode.get_by_uuid(graph_driver, node.uuid)
    assert retrieved.metadata == metadata
    assert retrieved.metadata['agent_id'] == 42
    assert retrieved.metadata['research_subject_id'] == 123


@pytest.mark.asyncio
async def test_entity_node_no_metadata(graph_driver):
    """EntityNode without metadata: retrieved metadata should be None or empty (backward compat)."""
    now = datetime.now()

    node = EntityNode(
        name='Bob',
        group_id=group_id,
        labels=['Entity'],
        created_at=now,
        summary='',
    )
    node.name_embedding = np.random.uniform(0, 1, 384).tolist()
    await node.save(graph_driver)

    retrieved = await EntityNode.get_by_uuid(graph_driver, node.uuid)
    # metadata should be None (no metadata stored)
    assert retrieved.metadata is None or retrieved.metadata == {}


@pytest.mark.asyncio
async def test_entity_node_empty_metadata(graph_driver):
    """EntityNode with empty metadata dict is stored without error."""
    now = datetime.now()

    node = EntityNode(
        name='Charlie',
        group_id=group_id,
        labels=['Entity'],
        created_at=now,
        summary='',
        metadata={},
    )
    node.name_embedding = np.random.uniform(0, 1, 384).tolist()
    await node.save(graph_driver)

    retrieved = await EntityNode.get_by_uuid(graph_driver, node.uuid)
    # empty dict stored; should come back empty or None
    assert retrieved.metadata is None or retrieved.metadata == {}


@pytest.mark.asyncio
async def test_entity_node_metadata_separate_from_attributes(graph_driver):
    """metadata and attributes are stored and retrieved independently."""
    now = datetime.now()
    metadata = {'user_provided': 'value'}
    attributes = {'age': 30, 'location': 'New York'}

    node = EntityNode(
        name='Dave',
        group_id=group_id,
        labels=['Entity'],
        created_at=now,
        summary='',
        attributes=attributes,
        metadata=metadata,
    )
    node.name_embedding = np.random.uniform(0, 1, 384).tolist()
    await node.save(graph_driver)

    retrieved = await EntityNode.get_by_uuid(graph_driver, node.uuid)
    assert retrieved.metadata == metadata
    assert 'user_provided' not in retrieved.attributes
    assert retrieved.attributes.get('age') == 30
    assert retrieved.attributes.get('location') == 'New York'


# ---------------------------------------------------------------------------
# EntityEdge metadata – save and retrieve
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_entity_edge_metadata_save_and_get(graph_driver):
    """EntityEdge.save() stores metadata; get_by_uuid() retrieves it."""
    now = datetime.now()
    metadata = {'session_id': 'abc123', 'agent_id': 7}

    source = EntityNode(
        name='NodeA', group_id=group_id, labels=['Entity'], created_at=now, summary=''
    )
    target = EntityNode(
        name='NodeB', group_id=group_id, labels=['Entity'], created_at=now, summary=''
    )
    source.name_embedding = np.random.uniform(0, 1, 384).tolist()
    target.name_embedding = np.random.uniform(0, 1, 384).tolist()
    await source.save(graph_driver)
    await target.save(graph_driver)

    edge = EntityEdge(
        source_node_uuid=source.uuid,
        target_node_uuid=target.uuid,
        name='RELATES_TO',
        fact='NodeA relates to NodeB',
        group_id=group_id,
        created_at=now,
        episodes=[],
        metadata=metadata,
    )
    edge.fact_embedding = np.random.uniform(0, 1, 384).tolist()
    await edge.save(graph_driver)

    retrieved = await EntityEdge.get_by_uuid(graph_driver, edge.uuid)
    assert retrieved.metadata == metadata
    assert retrieved.metadata['session_id'] == 'abc123'
    assert retrieved.metadata['agent_id'] == 7


@pytest.mark.asyncio
async def test_entity_edge_no_metadata(graph_driver):
    """EntityEdge without metadata: retrieved metadata is None or empty."""
    now = datetime.now()

    source = EntityNode(
        name='NodeC', group_id=group_id, labels=['Entity'], created_at=now, summary=''
    )
    target = EntityNode(
        name='NodeD', group_id=group_id, labels=['Entity'], created_at=now, summary=''
    )
    source.name_embedding = np.random.uniform(0, 1, 384).tolist()
    target.name_embedding = np.random.uniform(0, 1, 384).tolist()
    await source.save(graph_driver)
    await target.save(graph_driver)

    edge = EntityEdge(
        source_node_uuid=source.uuid,
        target_node_uuid=target.uuid,
        name='RELATES_TO',
        fact='NodeC relates to NodeD',
        group_id=group_id,
        created_at=now,
        episodes=[],
    )
    edge.fact_embedding = np.random.uniform(0, 1, 384).tolist()
    await edge.save(graph_driver)

    retrieved = await EntityEdge.get_by_uuid(graph_driver, edge.uuid)
    assert retrieved.metadata is None or retrieved.metadata == {}


# ---------------------------------------------------------------------------
# Bulk save with metadata
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_entity_node_metadata_bulk_save(graph_driver):
    """Bulk save propagates metadata to EntityNode."""
    if graph_driver.provider == GraphProvider.FALKORDB:
        pytest.skip('Skipping on FalkorDB in this suite')

    now = datetime.now()
    metadata = {'bulk_key': 'bulk_value', 'num': 99}
    embedder = _make_embedder()

    episode = EpisodicNode(
        name='bulk_ep',
        group_id=group_id,
        labels=[],
        created_at=now,
        source=EpisodeType.message,
        source_description='test',
        content='bulk test',
        valid_at=now,
        entity_edges=[],
    )

    node = EntityNode(
        name='BulkNode',
        group_id=group_id,
        labels=['Entity'],
        created_at=now,
        summary='',
        metadata=metadata,
    )

    await add_nodes_and_edges_bulk(
        driver=graph_driver,
        episodic_nodes=[episode],
        episodic_edges=[],
        entity_nodes=[node],
        entity_edges=[],
        embedder=embedder,
    )

    retrieved = await EntityNode.get_by_uuid(graph_driver, node.uuid)
    assert retrieved.metadata == metadata


@pytest.mark.asyncio
async def test_entity_edge_metadata_bulk_save(graph_driver):
    """Bulk save propagates metadata to EntityEdge."""
    if graph_driver.provider == GraphProvider.FALKORDB:
        pytest.skip('Skipping on FalkorDB in this suite')

    now = datetime.now()
    metadata = {'bulk_session': 'xyz', 'priority': 1}
    embedder = _make_embedder()

    episode = EpisodicNode(
        name='bulk_ep2',
        group_id=group_id,
        labels=[],
        created_at=now,
        source=EpisodeType.message,
        source_description='test',
        content='bulk edge test',
        valid_at=now,
        entity_edges=[],
    )

    source = EntityNode(
        name='BulkSrc', group_id=group_id, labels=['Entity'], created_at=now, summary=''
    )
    target = EntityNode(
        name='BulkTgt', group_id=group_id, labels=['Entity'], created_at=now, summary=''
    )

    edge = EntityEdge(
        source_node_uuid=source.uuid,
        target_node_uuid=target.uuid,
        name='BULK_REL',
        fact='BulkSrc bulk relates BulkTgt',
        group_id=group_id,
        created_at=now,
        episodes=[episode.uuid],
        metadata=metadata,
    )

    episode.entity_edges = [edge.uuid]

    episodic_edge = EpisodicEdge(
        source_node_uuid=episode.uuid,
        target_node_uuid=source.uuid,
        created_at=now,
        group_id=group_id,
    )

    await add_nodes_and_edges_bulk(
        driver=graph_driver,
        episodic_nodes=[episode],
        episodic_edges=[episodic_edge],
        entity_nodes=[source, target],
        entity_edges=[edge],
        embedder=embedder,
    )

    retrieved_edge = await EntityEdge.get_by_uuid(graph_driver, edge.uuid)
    assert retrieved_edge.metadata == metadata
    assert retrieved_edge.metadata['bulk_session'] == 'xyz'


# ---------------------------------------------------------------------------
# Metadata propagation from EpisodicNode to extracted nodes (unit-level)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_create_entity_nodes_inherits_episode_metadata():
    """_create_entity_nodes() assigns the episode's metadata to new EntityNode objects."""
    from graphiti_core.nodes import (
        EpisodeType,
        EpisodicNode,
    )
    from graphiti_core.utils.maintenance.node_operations import _create_entity_nodes
    from graphiti_core.prompts.extract_nodes import ExtractedEntity

    now = datetime.now()
    episode_meta = {'agent_id': 55, 'subject': 'test'}
    episode = EpisodicNode(
        name='ep',
        group_id=group_id,
        labels=[],
        created_at=now,
        source=EpisodeType.message,
        source_description='test',
        content='Alice is a person',
        valid_at=now,
        entity_edges=[],
        episode_metadata=episode_meta,
    )

    extracted = [
        ExtractedEntity(name='Alice', entity_type_id=0, episode_indices=[0]),
    ]
    entity_types_context = [
        {'entity_type_id': 0, 'entity_type_name': 'Entity', 'entity_type_description': 'A thing'}
    ]

    nodes, _ = _create_entity_nodes(extracted, entity_types_context, None, [episode])

    assert len(nodes) == 1
    assert nodes[0].metadata == episode_meta
    assert nodes[0].metadata['agent_id'] == 55
