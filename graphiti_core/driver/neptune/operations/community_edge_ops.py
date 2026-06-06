"""
Copyright 2024, Zep Software, Inc.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import logging
from typing import Any

from graphiti_core.driver.driver import GraphProvider
from graphiti_core.driver.operations.community_edge_ops import CommunityEdgeOperations
from graphiti_core.driver.query_executor import (
    QueryExecutor,
    Transaction,
)
from graphiti_core.edges import (
    CommunityEdge,
    get_community_edge_from_record,
)
from graphiti_core.errors import EdgeNotFoundError
from graphiti_core.models.edges.edge_db_queries import (
    get_community_edge_return_query,
    get_community_edge_save_query,
)

logger = logging.getLogger(__name__)


def _community_edge_data(edge: CommunityEdge) -> dict[str, Any]:
    edge_data: dict[str, Any] = {
        'uuid': edge.uuid,
        'group_id': edge.group_id,
        'created_at': edge.created_at,
    }
    for k, v in (edge.metadata or {}).items():
        edge_data[f'metadata_{k}'] = v
    return edge_data


class NeptuneCommunityEdgeOperations(CommunityEdgeOperations):
    async def save(
        self,
        executor: QueryExecutor,
        edge: CommunityEdge,
        tx: Transaction | None = None,
    ) -> None:
        query = get_community_edge_save_query(GraphProvider.NEPTUNE)
        params: dict[str, Any] = {
            'community_uuid': edge.source_node_uuid,
            'entity_uuid': edge.target_node_uuid,
            'edge_data': _community_edge_data(edge),
        }
        if tx is not None:
            await tx.run(query, **params)
        else:
            await executor.execute_query(query, **params)

        logger.debug(f'Saved Edge to Graph: {edge.uuid}')

    async def delete(
        self,
        executor: QueryExecutor,
        edge: CommunityEdge,
        tx: Transaction | None = None,
    ) -> None:
        query = """
            MATCH (n)-[e:MENTIONS|RELATES_TO|HAS_MEMBER {uuid: $uuid}]->(m)
            DELETE e
        """
        if tx is not None:
            await tx.run(query, uuid=edge.uuid)
        else:
            await executor.execute_query(query, uuid=edge.uuid)

        logger.debug(f'Deleted Edge: {edge.uuid}')

    async def delete_by_uuids(
        self,
        executor: QueryExecutor,
        uuids: list[str],
        tx: Transaction | None = None,
    ) -> None:
        query = """
            MATCH (n)-[e:MENTIONS|RELATES_TO|HAS_MEMBER]->(m)
            WHERE e.uuid IN $uuids
            DELETE e
        """
        if tx is not None:
            await tx.run(query, uuids=uuids)
        else:
            await executor.execute_query(query, uuids=uuids)

    async def get_by_uuid(
        self,
        executor: QueryExecutor,
        uuid: str,
    ) -> CommunityEdge:
        query = (
            """
            MATCH (n:Community)-[e:HAS_MEMBER {uuid: $uuid}]->(m)
            RETURN
            """
            + get_community_edge_return_query(GraphProvider.NEPTUNE)
        )
        records, _, _ = await executor.execute_query(query, uuid=uuid)
        edges = [get_community_edge_from_record(r) for r in records]
        if len(edges) == 0:
            raise EdgeNotFoundError(uuid)
        return edges[0]

    async def get_by_uuids(
        self,
        executor: QueryExecutor,
        uuids: list[str],
    ) -> list[CommunityEdge]:
        query = (
            """
            MATCH (n:Community)-[e:HAS_MEMBER]->(m)
            WHERE e.uuid IN $uuids
            RETURN
            """
            + get_community_edge_return_query(GraphProvider.NEPTUNE)
        )
        records, _, _ = await executor.execute_query(query, uuids=uuids)
        return [get_community_edge_from_record(r) for r in records]

    async def get_by_group_ids(
        self,
        executor: QueryExecutor,
        group_ids: list[str],
        limit: int | None = None,
        uuid_cursor: str | None = None,
    ) -> list[CommunityEdge]:
        cursor_clause = 'AND e.uuid < $uuid' if uuid_cursor else ''
        limit_clause = 'LIMIT $limit' if limit is not None else ''
        query = (
            """
            MATCH (n:Community)-[e:HAS_MEMBER]->(m)
            WHERE e.group_id IN $group_ids
            """
            + cursor_clause
            + """
            RETURN
            """
            + get_community_edge_return_query(GraphProvider.NEPTUNE)
            + """
            ORDER BY e.uuid DESC
            """
            + limit_clause
        )
        records, _, _ = await executor.execute_query(
            query,
            group_ids=group_ids,
            uuid=uuid_cursor,
            limit=limit,
        )
        return [get_community_edge_from_record(r) for r in records]
