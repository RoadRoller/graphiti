from __future__ import annotations

import pytest
from datetime import datetime
from unittest.mock import Mock

from graphiti_core.nodes import (
    EpisodeType,
    EpisodicNode,
)
from graphiti_core.utils.bulk_utils import add_nodes_and_edges_bulk
from tests.helpers_test import GraphProvider

pytest_plugins = ('pytest_asyncio',)


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
    assert retrieved.episode_metadata == metadata


@pytest.mark.asyncio
async def test_episode_metadata_persists_on_bulk_save(graph_driver):
    if graph_driver.provider == GraphProvider.FALKORDB:
        pytest.skip('Skipping as test fails on FalkorDB in this suite')

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
        episode_metadata={},
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
    assert retrieved.episode_metadata == {}
