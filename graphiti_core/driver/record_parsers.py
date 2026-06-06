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

import json
from typing import Any

from graphiti_core.edges import EntityEdge
from graphiti_core.helpers import parse_db_date
from graphiti_core.nodes import (
    CommunityNode,
    EntityNode,
    EpisodeType,
    EpisodicNode,
)


def _extract_metadata_from_attributes(attributes: dict[str, Any]) -> dict[str, Any] | None:
    """Extract metadata_* prefixed keys from an attributes dict, removing them in place."""
    metadata_keys = [k for k in list(attributes.keys()) if k.startswith('metadata_')]
    if not metadata_keys:
        return None
    metadata: dict[str, Any] = {}
    for key in metadata_keys:
        metadata[key[9:]] = attributes.pop(key)
    return metadata


def entity_node_from_record(record: Any) -> EntityNode:
    """Parse an entity node from a database record."""
    attributes = record['attributes']
    attributes.pop('uuid', None)
    attributes.pop('name', None)
    attributes.pop('group_id', None)
    attributes.pop('name_embedding', None)
    attributes.pop('summary', None)
    attributes.pop('created_at', None)
    attributes.pop('labels', None)

    metadata = _extract_metadata_from_attributes(attributes)

    labels = record.get('labels', [])
    group_id = record.get('group_id')
    dynamic_label = 'Entity_' + group_id.replace('-', '')
    if dynamic_label in labels:
        labels.remove(dynamic_label)

    return EntityNode(
        uuid=record['uuid'],
        name=record['name'],
        name_embedding=record.get('name_embedding'),
        group_id=group_id,
        labels=labels,
        created_at=parse_db_date(record['created_at']),  # type: ignore[arg-type]
        summary=record['summary'],
        attributes=attributes,
        metadata=metadata,
    )


def entity_edge_from_record(record: Any) -> EntityEdge:
    """Parse an entity edge from a database record."""
    attributes = record['attributes']
    attributes.pop('uuid', None)
    attributes.pop('source_node_uuid', None)
    attributes.pop('target_node_uuid', None)
    attributes.pop('fact', None)
    attributes.pop('fact_embedding', None)
    attributes.pop('name', None)
    attributes.pop('group_id', None)
    attributes.pop('episodes', None)
    attributes.pop('created_at', None)
    attributes.pop('expired_at', None)
    attributes.pop('valid_at', None)
    attributes.pop('invalid_at', None)
    attributes.pop('reference_time', None)

    metadata = _extract_metadata_from_attributes(attributes)

    return EntityEdge(
        uuid=record['uuid'],
        source_node_uuid=record['source_node_uuid'],
        target_node_uuid=record['target_node_uuid'],
        fact=record['fact'],
        fact_embedding=record.get('fact_embedding'),
        name=record['name'],
        group_id=record['group_id'],
        episodes=record['episodes'],
        created_at=parse_db_date(record['created_at']),  # type: ignore[arg-type]
        expired_at=parse_db_date(record['expired_at']),
        valid_at=parse_db_date(record['valid_at']),
        invalid_at=parse_db_date(record['invalid_at']),
        reference_time=parse_db_date(record.get('reference_time')),
        attributes=attributes,
        metadata=metadata,
    )


def episodic_node_from_record(record: Any) -> EpisodicNode:
    """Parse an episodic node from a database record."""
    created_at = parse_db_date(record['created_at'])
    valid_at = parse_db_date(record['valid_at'])

    if created_at is None:
        raise ValueError(f'created_at cannot be None for episode {record.get("uuid", "unknown")}')
    if valid_at is None:
        raise ValueError(f'valid_at cannot be None for episode {record.get("uuid", "unknown")}')

    # Support both flat metadata_* props (Neo4j/FalkorDB via ep_properties) and
    # JSON episode_metadata column (Kuzu/Neptune).
    # NOTE: use .get() not 'key in record' because neo4j Record.__contains__ checks VALUES.
    ep_props_raw = record.get('ep_properties')
    if ep_props_raw is not None:
        ep_props: dict[str, Any] = dict(ep_props_raw)
        metadata_keys = [k for k in ep_props if k.startswith('metadata_')]
        episode_metadata: dict[str, Any] | None = (
            {k[9:]: ep_props[k] for k in metadata_keys} if metadata_keys else None
        )
    else:
        raw_metadata = record.get('episode_metadata')
        episode_metadata = None
        if raw_metadata is not None:
            if isinstance(raw_metadata, dict):
                episode_metadata = raw_metadata
            elif isinstance(raw_metadata, str) and raw_metadata != '':
                try:
                    parsed = json.loads(raw_metadata)
                    if isinstance(parsed, dict):
                        episode_metadata = parsed
                except json.JSONDecodeError:
                    pass

    return EpisodicNode(
        content=record['content'],
        created_at=created_at,
        valid_at=valid_at,
        uuid=record['uuid'],
        group_id=record['group_id'],
        source=EpisodeType.from_str(record['source']),
        episode_metadata=episode_metadata,
        name=record['name'],
        source_description=record['source_description'],
        entity_edges=record['entity_edges'],
    )


def community_node_from_record(record: Any) -> CommunityNode:
    """Parse a community node from a database record."""
    return CommunityNode(
        uuid=record['uuid'],
        name=record['name'],
        group_id=record['group_id'],
        name_embedding=record['name_embedding'],
        created_at=parse_db_date(record['created_at']),  # type: ignore[arg-type]
        summary=record['summary'],
    )
