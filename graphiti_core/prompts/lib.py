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

Central registry for all LLM prompts used in Graphiti's ingestion and maintenance pipeline.

Extraction pipeline philosophy:
- **Structured KG mode** (standalone): ``extract_nodes`` + ``extract_edges`` via node/edge operations.
  Conservative entity policy — "when in doubt, do NOT extract" — suited for clean knowledge graphs.
- **Agent memory mode** (combined): ``extract_nodes_and_edges`` in a single LLM call.
  Liberal fact extraction — "when in doubt, extract the fact" — optimized for retrieval when
  original messages are not available at query time.

Choose the pipeline that matches your use case; mixing both on the same graph may produce
inconsistent entity granularity.
"""

from typing import (
    Any,
    Protocol,
    TypedDict,
)

from .dedupe_edges import (
    Prompt as DedupeEdgesPrompt,
    Versions as DedupeEdgesVersions,
    versions as dedupe_edges_versions,
)
from .dedupe_nodes import (
    Prompt as DedupeNodesPrompt,
    Versions as DedupeNodesVersions,
    versions as dedupe_nodes_versions,
)
from .eval import (
    Prompt as EvalPrompt,
    Versions as EvalVersions,
    versions as eval_versions,
)
from .extract_edges import (
    Prompt as ExtractEdgesPrompt,
    Versions as ExtractEdgesVersions,
    versions as extract_edges_versions,
)
from .extract_nodes import (
    Prompt as ExtractNodesPrompt,
    Versions as ExtractNodesVersions,
    versions as extract_nodes_versions,
)
from .extract_nodes_and_edges import (
    Prompt as ExtractNodesAndEdgesPrompt,
    Versions as ExtractNodesAndEdgesVersions,
    versions as extract_nodes_and_edges_versions,
)
from .models import (
    Message,
    PromptFunction,
)
from .prompt_helpers import DO_NOT_ESCAPE_UNICODE
from .summarize_nodes import (
    Prompt as SummarizeNodesPrompt,
    Versions as SummarizeNodesVersions,
    versions as summarize_nodes_versions,
)
from .summarize_sagas import (
    Prompt as SummarizeSagasPrompt,
    Versions as SummarizeSagasVersions,
    versions as summarize_sagas_versions,
)


class PromptLibrary(Protocol):
    extract_nodes: ExtractNodesPrompt
    dedupe_nodes: DedupeNodesPrompt
    extract_edges: ExtractEdgesPrompt
    extract_nodes_and_edges: ExtractNodesAndEdgesPrompt
    dedupe_edges: DedupeEdgesPrompt
    summarize_nodes: SummarizeNodesPrompt
    summarize_sagas: SummarizeSagasPrompt
    eval: EvalPrompt


class PromptLibraryImpl(TypedDict):
    extract_nodes: ExtractNodesVersions
    dedupe_nodes: DedupeNodesVersions
    extract_edges: ExtractEdgesVersions
    extract_nodes_and_edges: ExtractNodesAndEdgesVersions
    dedupe_edges: DedupeEdgesVersions
    summarize_nodes: SummarizeNodesVersions
    summarize_sagas: SummarizeSagasVersions
    eval: EvalVersions


class VersionWrapper:
    def __init__(self, func: PromptFunction):
        self.func = func

    def __call__(self, context: dict[str, Any]) -> list[Message]:
        messages = self.func(context)
        for message in messages:
            message.content += DO_NOT_ESCAPE_UNICODE if message.role == 'system' else ''
        return messages


class PromptTypeWrapper:
    def __init__(self, versions: dict[str, PromptFunction]):
        for version, func in versions.items():
            setattr(self, version, VersionWrapper(func))


class PromptLibraryWrapper:
    def __init__(self, library: PromptLibraryImpl):
        for prompt_type, versions in library.items():
            setattr(self, prompt_type, PromptTypeWrapper(versions))  # type: ignore[arg-type]


PROMPT_LIBRARY_IMPL: PromptLibraryImpl = {
    'extract_nodes': extract_nodes_versions,
    'dedupe_nodes': dedupe_nodes_versions,
    'extract_edges': extract_edges_versions,
    'extract_nodes_and_edges': extract_nodes_and_edges_versions,
    'dedupe_edges': dedupe_edges_versions,
    'summarize_nodes': summarize_nodes_versions,
    'summarize_sagas': summarize_sagas_versions,
    'eval': eval_versions,
}
prompt_library: PromptLibrary = PromptLibraryWrapper(PROMPT_LIBRARY_IMPL)  # type: ignore[assignment]
