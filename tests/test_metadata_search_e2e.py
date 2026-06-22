"""
End-to-end database tests for search with metadata filtering.

Exercises the full search pipeline (index lookup + Cypher filter) against a
real graph driver, not just query constructor unit tests.
"""

from __future__ import annotations

import numpy as np
import pytest
from datetime import datetime
from unittest.mock import (
    AsyncMock,
    Mock,
)

from graphiti_core.cross_encoder.client import CrossEncoderClient
from graphiti_core.edges import EntityEdge
from graphiti_core.embedder.client import EmbedderClient
from graphiti_core.graphiti import Graphiti
from graphiti_core.llm_client import LLMClient
from graphiti_core.nodes import (
    EntityNode,
    EpisodeType,
    EpisodicNode,
)
from graphiti_core.search.search_config_recipes import COMBINED_HYBRID_SEARCH_RRF
from graphiti_core.search.search_filters import SearchFilters
from graphiti_core.search.search_utils import (
    edge_fulltext_search,
    episode_fulltext_search,
    node_fulltext_search,
)
from tests.helpers_test import (
    GraphProvider,
    group_id,
)

pytest_plugins = ('pytest_asyncio',)


def _make_embedder():
    mock = Mock(spec=EmbedderClient)
    rng = np.random.default_rng(42)

    async def _async_create(input_data):
        return rng.uniform(0, 1, 384).tolist()

    mock.create = _async_create
    return mock


@pytest.fixture
def local_embedder():
    return _make_embedder()


@pytest.fixture
def mock_llm_client():
    mock_llm = Mock(spec=LLMClient)
    mock_llm.config = Mock()
    mock_llm.model = 'test-model'
    mock_llm.small_model = 'test-small-model'
    mock_llm.temperature = 0.0
    mock_llm.max_tokens = 1000
    mock_llm.cache_enabled = False
    mock_llm.cache_dir = None
    mock_llm.generate_response = AsyncMock()
    return mock_llm


@pytest.fixture
def mock_cross_encoder_client():
    mock_ce = Mock(spec=CrossEncoderClient)
    mock_ce.config = Mock()
    mock_ce.rerank = AsyncMock(return_value=[])
    return mock_ce


@pytest.fixture
async def indices_built(graph_driver):
    if graph_driver.provider == GraphProvider.FALKORDB:
        pytest.skip('Skipping as tests fail on FalkorDB in this suite')

    await graph_driver.build_indices_and_constraints()


@pytest.fixture
async def graphiti_with_indices(
        graph_driver, local_embedder, mock_llm_client, mock_cross_encoder_client
):
    if graph_driver.provider == GraphProvider.FALKORDB:
        pytest.skip('Skipping as tests fail on FalkorDB in this suite')

    graphiti = Graphiti(
        graph_driver=graph_driver,
        embedder=local_embedder,
        llm_client=mock_llm_client,
        cross_encoder=mock_cross_encoder_client,
    )
    await graphiti.build_indices_and_constraints()
    yield graphiti
    await graphiti.close()


async def _save_searchable_nodes(graph_driver, embedder):
    now = datetime.now()

    node_agent_1 = EntityNode(
        name='metadata_filter_agent_one',
        group_id=group_id,
        labels=['Entity'],
        created_at=now,
        summary='Researcher working on quantum mechanics',
        metadata={'agent_id': 1, 'research_subject_id': 100},
    )
    node_agent_2 = EntityNode(
        name='metadata_filter_agent_two',
        group_id=group_id,
        labels=['Entity'],
        created_at=now,
        summary='Researcher working on quantum mechanics',
        metadata={'agent_id': 2, 'research_subject_id': 100},
    )

    await node_agent_1.generate_name_embedding(embedder)
    await node_agent_2.generate_name_embedding(embedder)
    await node_agent_1.save(graph_driver)
    await node_agent_2.save(graph_driver)

    return node_agent_1, node_agent_2


async def _save_searchable_edges(graph_driver, embedder):
    now = datetime.now()

    source_a = EntityNode(
        name='metadata_edge_source_alpha',
        group_id=group_id,
        labels=['Entity'],
        created_at=now,
        summary='',
    )
    target_a = EntityNode(
        name='metadata_edge_target_alpha',
        group_id=group_id,
        labels=['Entity'],
        created_at=now,
        summary='',
    )
    source_b = EntityNode(
        name='metadata_edge_source_beta',
        group_id=group_id,
        labels=['Entity'],
        created_at=now,
        summary='',
    )
    target_b = EntityNode(
        name='metadata_edge_target_beta',
        group_id=group_id,
        labels=['Entity'],
        created_at=now,
        summary='',
    )

    for node in (source_a, target_a, source_b, target_b):
        await node.generate_name_embedding(embedder)
        await node.save(graph_driver)

    edge_session_abc = EntityEdge(
        source_node_uuid=source_a.uuid,
        target_node_uuid=target_a.uuid,
        name='WORKS_AT',
        fact='metadata_edge_source_alpha works at metadata_edge_target_alpha',
        group_id=group_id,
        created_at=now,
        episodes=[],
        metadata={'session_id': 'abc', 'agent_id': 1},
    )
    edge_session_xyz = EntityEdge(
        source_node_uuid=source_b.uuid,
        target_node_uuid=target_b.uuid,
        name='WORKS_AT',
        fact='metadata_edge_source_beta works at metadata_edge_target_beta',
        group_id=group_id,
        created_at=now,
        episodes=[],
        metadata={'session_id': 'xyz', 'agent_id': 2},
    )

    await edge_session_abc.generate_embedding(embedder)
    await edge_session_xyz.generate_embedding(embedder)
    await edge_session_abc.save(graph_driver)
    await edge_session_xyz.save(graph_driver)

    return edge_session_abc, edge_session_xyz


@pytest.mark.asyncio
async def test_node_fulltext_search_filters_by_single_metadata_field(
        graph_driver, local_embedder, indices_built
):
    await _save_searchable_nodes(graph_driver, local_embedder)

    results = await node_fulltext_search(
        graph_driver,
        'quantum mechanics',
        SearchFilters(metadata={'agent_id': 1}),
        group_ids=[group_id],
    )

    assert len(results) == 1
    assert results[0].name == 'metadata_filter_agent_one'
    assert results[0].metadata is not None
    assert results[0].metadata['agent_id'] == 1


@pytest.mark.asyncio
async def test_node_fulltext_search_filters_by_multiple_metadata_fields(
        graph_driver, local_embedder, indices_built
):
    await _save_searchable_nodes(graph_driver, local_embedder)

    results = await node_fulltext_search(
        graph_driver,
        'quantum mechanics',
        SearchFilters(metadata={'agent_id': 1, 'research_subject_id': 100}),
        group_ids=[group_id],
    )

    assert len(results) == 1
    assert results[0].metadata is not None
    assert results[0].metadata['agent_id'] == 1
    assert results[0].metadata['research_subject_id'] == 100


@pytest.mark.asyncio
async def test_node_fulltext_search_without_metadata_returns_all(
        graph_driver, local_embedder, indices_built
):
    await _save_searchable_nodes(graph_driver, local_embedder)

    results = await node_fulltext_search(
        graph_driver,
        'quantum mechanics',
        SearchFilters(),
        group_ids=[group_id],
    )

    agent_ids = {node.metadata['agent_id'] for node in results if node.metadata}
    assert agent_ids == {1, 2}


@pytest.mark.asyncio
async def test_edge_fulltext_search_filters_by_metadata(
        graph_driver, local_embedder, indices_built
):
    edge_abc, _ = await _save_searchable_edges(graph_driver, local_embedder)

    results = await edge_fulltext_search(
        graph_driver,
        'works at',
        SearchFilters(metadata={'session_id': 'abc'}),
        group_ids=[group_id],
    )

    assert len(results) == 1
    assert results[0].uuid == edge_abc.uuid
    assert results[0].metadata is not None
    assert results[0].metadata['session_id'] == 'abc'


@pytest.mark.asyncio
async def test_graphiti_search_filters_edges_by_metadata(
        graph_driver, local_embedder, graphiti_with_indices
):
    edge_abc, edge_xyz = await _save_searchable_edges(graph_driver, local_embedder)

    results = await graphiti_with_indices.search(
        query='works at',
        group_ids=[group_id],
        search_filter=SearchFilters(metadata={'agent_id': 1}),
    )

    assert len(results) == 1
    assert results[0].uuid == edge_abc.uuid
    assert results[0].metadata is not None
    assert results[0].metadata['agent_id'] == 1

    all_results = await graphiti_with_indices.search(
        query='works at',
        group_ids=[group_id],
        search_filter=SearchFilters(),
    )
    result_uuids = {edge.uuid for edge in all_results}
    assert edge_abc.uuid in result_uuids
    assert edge_xyz.uuid in result_uuids


@pytest.mark.asyncio
async def test_episode_fulltext_search_filters_by_metadata(graph_driver, indices_built):
    if graph_driver.provider == GraphProvider.KUZU:
        pytest.skip('Skipping as fulltext indexing not supported for Kuzu')

    now = datetime.now()
    metadata = {'research_subject_id': 34, 'agent_id': 1}

    episode = EpisodicNode(
        name='scoped_episode',
        group_id=group_id,
        labels=[],
        created_at=now,
        source=EpisodeType.message,
        source_description='metadata scope test',
        content='unique scoped episode searchable content',
        valid_at=now,
        entity_edges=[],
        episode_metadata=metadata,
    )
    await episode.save(graph_driver)

    results = await episode_fulltext_search(
        graph_driver,
        'scoped episode searchable',
        SearchFilters(metadata={'research_subject_id': 34}),
        group_ids=[group_id],
    )

    assert len(results) == 1
    assert results[0].uuid == episode.uuid


@pytest.mark.asyncio
async def test_episode_fulltext_search_with_non_matching_metadata_returns_empty(
        graph_driver, indices_built
):
    if graph_driver.provider == GraphProvider.KUZU:
        pytest.skip('Skipping as fulltext indexing not supported for Kuzu')

    now = datetime.now()
    episode = EpisodicNode(
        name='unscoped_episode',
        group_id=group_id,
        labels=[],
        created_at=now,
        source=EpisodeType.message,
        source_description='metadata scope test',
        content='unique unscoped episode searchable content',
        valid_at=now,
        entity_edges=[],
        episode_metadata={'research_subject_id': 34},
    )
    await episode.save(graph_driver)

    results = await episode_fulltext_search(
        graph_driver,
        'unscoped episode searchable',
        SearchFilters(metadata={'research_subject_id': 99}),
        group_ids=[group_id],
    )

    assert results == []


@pytest.mark.asyncio
async def test_edge_fulltext_search_filters_by_linked_episode_metadata(
        graph_driver, local_embedder, indices_built
):
    now = datetime.now()
    episode = EpisodicNode(
        name='edge_scope_episode',
        group_id=group_id,
        labels=[],
        created_at=now,
        source=EpisodeType.message,
        source_description='edge episode linkage test',
        content='edge scope episode content',
        valid_at=now,
        entity_edges=[],
        episode_metadata={'research_subject_id': 34},
    )
    await episode.save(graph_driver)

    source = EntityNode(
        name='edge_scope_source',
        group_id=group_id,
        labels=['Entity'],
        created_at=now,
        summary='',
    )
    target = EntityNode(
        name='edge_scope_target',
        group_id=group_id,
        labels=['Entity'],
        created_at=now,
        summary='',
    )
    for node in (source, target):
        await node.generate_name_embedding(local_embedder)
        await node.save(graph_driver)

    edge = EntityEdge(
        source_node_uuid=source.uuid,
        target_node_uuid=target.uuid,
        name='WORKS_AT',
        fact='edge_scope_source works at edge_scope_target',
        group_id=group_id,
        created_at=now,
        episodes=[episode.uuid],
        metadata=None,
    )
    await edge.generate_embedding(local_embedder)
    await edge.save(graph_driver)

    results = await edge_fulltext_search(
        graph_driver,
        'works at',
        SearchFilters(metadata={'research_subject_id': 34}),
        group_ids=[group_id],
    )

    assert len(results) == 1
    assert results[0].uuid == edge.uuid


@pytest.mark.asyncio
async def test_graphiti_search_with_metadata_scope_returns_empty_for_missing_scope(
        graph_driver, local_embedder, graphiti_with_indices
):
    now = datetime.now()
    episode = EpisodicNode(
        name='combined_scope_episode',
        group_id=group_id,
        labels=[],
        created_at=now,
        source=EpisodeType.message,
        source_description='combined search scope test',
        content='combined scope searchable episode content',
        valid_at=now,
        entity_edges=[],
        episode_metadata={'research_subject_id': 34},
    )
    await episode.save(graph_driver)

    source = EntityNode(
        name='combined_scope_source',
        group_id=group_id,
        labels=['Entity'],
        created_at=now,
        summary='combined scope researcher',
    )
    await source.generate_name_embedding(local_embedder)
    await source.save(graph_driver)

    edge = EntityEdge(
        source_node_uuid=source.uuid,
        target_node_uuid=source.uuid,
        name='RELATED_TO',
        fact='combined scope source self relation',
        group_id=group_id,
        created_at=now,
        episodes=[episode.uuid],
        metadata=None,
    )
    await edge.generate_embedding(local_embedder)
    await edge.save(graph_driver)

    search_results = await graphiti_with_indices.search_(
        query='combined scope',
        group_ids=[group_id],
        config=COMBINED_HYBRID_SEARCH_RRF,
        search_filter=SearchFilters(metadata={'research_subject_id': 99}),
    )

    assert search_results.episodes == []
    assert search_results.edges == []
    assert search_results.nodes == []
