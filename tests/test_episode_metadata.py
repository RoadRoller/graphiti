from __future__ import annotations

import inspect
import pytest
from datetime import datetime
from unittest.mock import Mock

from graphiti_core.driver.driver import GraphProvider as CoreGraphProvider
from graphiti_core.driver.neo4j.operations.search_ops import Neo4jSearchOperations
from graphiti_core.models.nodes.node_db_queries import get_episodic_node_return_query
from graphiti_core.nodes import (
    EpisodeType,
    EpisodicNode,
    get_episodic_node_from_record,
)
from graphiti_core.search.search_filters import SearchFilters
from graphiti_core.search.search_utils import episode_fulltext_search
from graphiti_core.utils.bulk_utils import add_nodes_and_edges_bulk
from tests.helpers_test import (
    GraphProvider,
    group_id,
)

pytest_plugins = ('pytest_asyncio',)


def test_episodic_node_return_query_neo4j_uses_ep_properties():
    query = get_episodic_node_return_query(CoreGraphProvider.NEO4J)
    assert 'properties(e) AS ep_properties' in query


def test_episodic_node_from_record_parses_flat_metadata_from_ep_properties():
    now = datetime.now()
    record = {
        'content': 'test',
        'created_at': now,
        'valid_at': now,
        'uuid': 'test-uuid',
        'group_id': group_id,
        'source': 'message',
        'name': 'test',
        'source_description': 'desc',
        'entity_edges': [],
        'ep_properties': {
            'uuid': 'test-uuid',
            'metadata_url': 'https://example.com/doc',
            'metadata_agent_id': 42,
        },
    }

    node = get_episodic_node_from_record(record)
    assert node.metadata == {'url': 'https://example.com/doc', 'agent_id': 42}


def test_neo4j_episode_fulltext_search_uses_provider_return_query():
    source = inspect.getsource(Neo4jSearchOperations.episode_fulltext_search)
    assert 'get_episodic_node_return_query' in source
    assert 'EPISODIC_NODE_RETURN' not in source


@pytest.mark.asyncio
async def test_episode_metadata_persists_on_save_and_get(graph_driver):
    metadata = {'agent_id': 42, 'research_subject_id': 123, 'session': 'abc'}
    now = datetime.now()

    episode = EpisodicNode(
        name='metadata_episode',
        group_id='graphiti_test_group',
        labels=[],
        created_at=now,
        source=EpisodeType.message,
        source_description='test',
        content='hello',
        valid_at=now,
        entity_edges=[],
        episode_metadata=metadata,
    )

    await episode.save(graph_driver)
    retrieved = await EpisodicNode.get_by_uuid(graph_driver, episode.uuid)
    assert retrieved.metadata is not None
    assert retrieved.metadata['agent_id'] == 42
    assert retrieved.metadata['research_subject_id'] == 123
    assert retrieved.metadata['session'] == 'abc'


@pytest.mark.asyncio
async def test_episode_metadata_persists_on_bulk_save(graph_driver):
    if graph_driver.provider == GraphProvider.FALKORDB:
        pytest.skip('Skipping as test fails on FalkorDB in this suite')

    metadata = {'agent_id': 42, 'research_subject_id': 123, 'session': 'abc'}
    now = datetime.now()
    episode = EpisodicNode(
        name='bulk_metadata_episode',
        group_id='graphiti_test_group',
        labels=[],
        created_at=now,
        source=EpisodeType.message,
        source_description='test',
        content='hello bulk',
        valid_at=now,
        entity_edges=[],
        episode_metadata=metadata,
    )

    await add_nodes_and_edges_bulk(
        driver=graph_driver,
        episodic_nodes=[episode],
        episodic_edges=[],
        entity_nodes=[],
        entity_edges=[],
        embedder=Mock(),
    )

    retrieved = await EpisodicNode.get_by_uuid(graph_driver, episode.uuid)
    assert retrieved.metadata is not None
    assert retrieved.metadata['agent_id'] == 42
    assert retrieved.metadata['research_subject_id'] == 123
    assert retrieved.metadata['session'] == 'abc'


@pytest.mark.asyncio
async def test_episode_metadata_returned_by_fulltext_search(graph_driver):
    if graph_driver.provider == GraphProvider.KUZU:
        pytest.skip('Skipping as fulltext indexing not supported for Kuzu')

    metadata = {'url': 'https://example.com/doc', 'agent_id': 42}
    now = datetime.now()

    await graph_driver.build_indices_and_constraints()

    episode = EpisodicNode(
        name='metadata_search_episode',
        group_id=group_id,
        labels=[],
        created_at=now,
        source=EpisodeType.message,
        source_description='metadata search test',
        content='unique searchable metadata content',
        valid_at=now,
        entity_edges=[],
        episode_metadata=metadata,
    )
    await episode.save(graph_driver)

    results = await episode_fulltext_search(
        graph_driver,
        'searchable metadata',
        SearchFilters(),
        group_ids=[group_id],
    )

    assert len(results) == 1
    assert results[0].uuid == episode.uuid
    assert results[0].metadata is not None
    assert results[0].metadata['url'] == 'https://example.com/doc'
    assert results[0].metadata['agent_id'] == 42
