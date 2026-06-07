"""
Tests for metadata on structural nodes and edges.

Covers SagaNode and edge types that are not EntityEdge: MENTIONS (EpisodicEdge),
HAS_EPISODE, and NEXT_EPISODE.
"""
from __future__ import annotations

import inspect
import numpy as np
import pytest
from datetime import datetime
from unittest.mock import Mock

from graphiti_core.driver.neo4j.operations.saga_node_ops import Neo4jSagaNodeOperations
from graphiti_core.edges import (
    CommunityEdge,
    EpisodicEdge,
    HasEpisodeEdge,
    NextEpisodeEdge,
)
from graphiti_core.models.nodes.node_db_queries import build_saga_save_data
from graphiti_core.nodes import (
    CommunityNode,
    EntityNode,
    EpisodeType,
    EpisodicNode,
    SagaNode,
    get_saga_node_from_record,
)
from graphiti_core.utils.bulk_utils import add_nodes_and_edges_bulk
from graphiti_core.utils.maintenance.edge_operations import build_community_edges
from tests.helpers_test import (
    GraphProvider,
    group_id,
)

pytest_plugins = ('pytest_asyncio',)

SAMPLE_METADATA = {
    'agent_id': 1,
    'session_id': 241,
    'research_subject_id': 34,
    'artifact_id': 146,
    'message_id': 1790,
}


def test_build_saga_save_data_flattens_metadata():
    now = datetime.now()
    saga_data = build_saga_save_data(
        uuid='saga-uuid',
        name='test_saga',
        group_id=group_id,
        created_at=now,
        metadata=SAMPLE_METADATA,
    )

    assert saga_data['metadata_agent_id'] == 1
    assert saga_data['metadata_session_id'] == 241
    assert saga_data['metadata_research_subject_id'] == 34


def test_saga_node_from_record_parses_flat_metadata_from_saga_properties():
    now = datetime.now()
    record = {
        'uuid': 'saga-uuid',
        'name': 'test_saga',
        'group_id': group_id,
        'created_at': now,
        'summary': '',
        'first_episode_uuid': None,
        'last_episode_uuid': None,
        'last_summarized_at': None,
        'last_summarized_episode_valid_at': None,
        'saga_properties': {
            'uuid': 'saga-uuid',
            'metadata_agent_id': 1,
            'metadata_session_id': 241,
        },
    }

    node = get_saga_node_from_record(record)
    assert node.metadata == {'agent_id': 1, 'session_id': 241}


def test_neo4j_saga_node_ops_save_uses_build_saga_save_data():
    source = inspect.getsource(Neo4jSagaNodeOperations.save)
    assert 'build_saga_save_data' in source
    assert 'saga_data=saga_data' in source


def _make_embedder():
    mock = Mock()
    rng = np.random.default_rng(0)

    async def _async_create(input_data):
        return rng.uniform(0, 1, 384).tolist()

    mock.create = _async_create
    return mock


@pytest.mark.asyncio
async def test_saga_node_metadata_save_and_get(graph_driver):
    now = datetime.now()
    saga = SagaNode(
        name='metadata_saga',
        group_id=group_id,
        created_at=now,
        metadata=SAMPLE_METADATA,
    )

    await saga.save(graph_driver)
    retrieved = await SagaNode.get_by_uuid(graph_driver, saga.uuid)

    assert retrieved.metadata == SAMPLE_METADATA
    assert retrieved.metadata['agent_id'] == 1
    assert retrieved.metadata['session_id'] == 241


@pytest.mark.asyncio
async def test_episodic_edge_mentions_metadata_save_and_get(graph_driver):
    now = datetime.now()

    episode = EpisodicNode(
        name='mentions_episode',
        group_id=group_id,
        labels=[],
        created_at=now,
        source=EpisodeType.message,
        source_description='test',
        content='hello',
        valid_at=now,
        entity_edges=[],
        metadata=SAMPLE_METADATA,
    )
    entity = EntityNode(
        name='MentionedEntity',
        group_id=group_id,
        labels=['Entity'],
        created_at=now,
        summary='',
    )
    entity.name_embedding = np.random.uniform(0, 1, 384).tolist()

    await episode.save(graph_driver)
    await entity.save(graph_driver)

    edge = EpisodicEdge(
        source_node_uuid=episode.uuid,
        target_node_uuid=entity.uuid,
        group_id=group_id,
        created_at=now,
        metadata=SAMPLE_METADATA,
    )
    await edge.save(graph_driver)

    retrieved = await EpisodicEdge.get_by_uuid(graph_driver, edge.uuid)
    assert retrieved.metadata == SAMPLE_METADATA


@pytest.mark.asyncio
async def test_has_episode_edge_metadata_save_and_get(graph_driver):
    now = datetime.now()

    saga = SagaNode(name='has_episode_saga', group_id=group_id, created_at=now)
    episode = EpisodicNode(
        name='has_episode_ep',
        group_id=group_id,
        labels=[],
        created_at=now,
        source=EpisodeType.message,
        source_description='test',
        content='saga episode',
        valid_at=now,
        entity_edges=[],
    )

    await saga.save(graph_driver)
    await episode.save(graph_driver)

    edge = HasEpisodeEdge(
        source_node_uuid=saga.uuid,
        target_node_uuid=episode.uuid,
        group_id=group_id,
        created_at=now,
        metadata=SAMPLE_METADATA,
    )
    await edge.save(graph_driver)

    retrieved = await HasEpisodeEdge.get_by_uuid(graph_driver, edge.uuid)
    assert retrieved.metadata == SAMPLE_METADATA


@pytest.mark.asyncio
async def test_next_episode_edge_metadata_save_and_get(graph_driver):
    now = datetime.now()

    episode_1 = EpisodicNode(
        name='next_episode_1',
        group_id=group_id,
        labels=[],
        created_at=now,
        source=EpisodeType.message,
        source_description='test',
        content='first',
        valid_at=now,
        entity_edges=[],
    )
    episode_2 = EpisodicNode(
        name='next_episode_2',
        group_id=group_id,
        labels=[],
        created_at=now,
        source=EpisodeType.message,
        source_description='test',
        content='second',
        valid_at=now,
        entity_edges=[],
    )

    await episode_1.save(graph_driver)
    await episode_2.save(graph_driver)

    edge = NextEpisodeEdge(
        source_node_uuid=episode_1.uuid,
        target_node_uuid=episode_2.uuid,
        group_id=group_id,
        created_at=now,
        metadata=SAMPLE_METADATA,
    )
    await edge.save(graph_driver)

    retrieved = await NextEpisodeEdge.get_by_uuid(graph_driver, edge.uuid)
    assert retrieved.metadata == SAMPLE_METADATA


@pytest.mark.asyncio
async def test_episodic_edge_metadata_bulk_save(graph_driver):
    if graph_driver.provider == GraphProvider.FALKORDB:
        pytest.skip('Skipping as test fails on FalkorDB in this suite')

    now = datetime.now()
    embedder = _make_embedder()

    episode = EpisodicNode(
        name='bulk_mentions_episode',
        group_id=group_id,
        labels=[],
        created_at=now,
        source=EpisodeType.message,
        source_description='test',
        content='bulk mentions',
        valid_at=now,
        entity_edges=[],
        metadata=SAMPLE_METADATA,
    )
    entity = EntityNode(
        name='BulkMentionedEntity',
        group_id=group_id,
        labels=['Entity'],
        created_at=now,
        summary='',
        metadata=SAMPLE_METADATA,
    )
    episodic_edge = EpisodicEdge(
        source_node_uuid=episode.uuid,
        target_node_uuid=entity.uuid,
        group_id=group_id,
        created_at=now,
        metadata=SAMPLE_METADATA,
    )

    await add_nodes_and_edges_bulk(
        driver=graph_driver,
        episodic_nodes=[episode],
        episodic_edges=[episodic_edge],
        entity_nodes=[entity],
        entity_edges=[],
        embedder=embedder,
    )

    retrieved = await EpisodicEdge.get_by_uuid(graph_driver, episodic_edge.uuid)
    assert retrieved.metadata == SAMPLE_METADATA


@pytest.mark.asyncio
async def test_community_edge_has_member_metadata_save_and_get(graph_driver):
    now = datetime.now()

    community = CommunityNode(
        name='metadata_community',
        group_id=group_id,
        labels=['Community'],
        created_at=now,
        summary='test community',
    )
    community.name_embedding = np.random.uniform(0, 1, 384).tolist()

    entity = EntityNode(
        name='CommunityMember',
        group_id=group_id,
        labels=['Entity'],
        created_at=now,
        summary='',
        metadata=SAMPLE_METADATA,
    )
    entity.name_embedding = np.random.uniform(0, 1, 384).tolist()

    await community.save(graph_driver)
    await entity.save(graph_driver)

    edge = CommunityEdge(
        source_node_uuid=community.uuid,
        target_node_uuid=entity.uuid,
        group_id=group_id,
        created_at=now,
        metadata=SAMPLE_METADATA,
    )
    await edge.save(graph_driver)

    retrieved = await CommunityEdge.get_by_uuid(graph_driver, edge.uuid)
    assert retrieved.metadata == SAMPLE_METADATA


@pytest.mark.asyncio
async def test_build_community_edges_inherits_entity_metadata():
    now = datetime.now()
    community = CommunityNode(
        name='build_community',
        group_id=group_id,
        labels=['Community'],
        created_at=now,
        summary='cluster',
    )
    entity = EntityNode(
        name='ClusterMember',
        group_id=group_id,
        labels=['Entity'],
        created_at=now,
        summary='',
        metadata=SAMPLE_METADATA,
    )

    edges = build_community_edges([entity], community, now)

    assert len(edges) == 1
    assert edges[0].metadata == SAMPLE_METADATA
    assert edges[0].group_id == group_id


@pytest.mark.asyncio
async def test_episode_metadata_stored_as_flat_properties(graph_driver):
    """Neo4j/FalkorDB store episode metadata as metadata_* flat properties, not JSON."""
    if graph_driver.provider not in (GraphProvider.NEO4J, GraphProvider.FALKORDB):
        pytest.skip('Flat metadata_* storage applies to Neo4j and FalkorDB')

    now = datetime.now()
    episode = EpisodicNode(
        name='flat_metadata_episode',
        group_id=group_id,
        labels=[],
        created_at=now,
        source=EpisodeType.message,
        source_description='test',
        content='flat storage check',
        valid_at=now,
        entity_edges=[],
        metadata=SAMPLE_METADATA,
    )
    await episode.save(graph_driver)

    records, _, _ = await graph_driver.execute_query(
        'MATCH (e:Episodic {uuid: $uuid}) RETURN properties(e) AS props',
        uuid=episode.uuid,
    )
    props = records[0]['props']

    assert props['metadata_agent_id'] == 1
    assert props['metadata_session_id'] == 241
    assert props['metadata_research_subject_id'] == 34
    assert 'episode_metadata' not in props
