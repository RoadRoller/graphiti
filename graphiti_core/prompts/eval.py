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

from pydantic import (
    BaseModel,
    Field,
)
from typing import (
    Any,
    Protocol,
    TypedDict,
)

from .models import (
    Message,
    PromptFunction,
    PromptVersion,
)
from .prompt_helpers import to_prompt_json


class QueryExpansion(BaseModel):
    query: str = Field(..., description='query optimized for database search')


class QAResponse(BaseModel):
    ANSWER: str = Field(..., description='how the subject would answer the question')


class EvalResponse(BaseModel):
    is_correct: bool = Field(..., description='boolean if the answer is correct or incorrect')
    reasoning: str = Field(
        ..., description='why you determined the response was correct or incorrect'
    )


class EvalAddEpisodeResults(BaseModel):
    candidate_is_worse: bool = Field(
        ...,
        description='boolean if the baseline extraction is higher quality than the candidate extraction.',
    )
    reasoning: str = Field(
        ..., description='why you determined the response was correct or incorrect'
    )


class Prompt(Protocol):
    qa_prompt: PromptVersion
    eval_prompt: PromptVersion
    query_expansion: PromptVersion
    eval_add_episode_results: PromptVersion


class Versions(TypedDict):
    qa_prompt: PromptFunction
    eval_prompt: PromptFunction
    query_expansion: PromptFunction
    eval_add_episode_results: PromptFunction


_GRAPH_QUALITY_RUBRIC = """
Evaluate graph extraction quality using these criteria (higher weight at the top):

1. **Entity specificity**: Entities are concrete and uniquely identifiable — not bare pronouns,
   generic nouns ("stuff", "event", "pic"), or sentence fragments. Possessive qualification
   used for relatives/pets ("Nisha's dad" not "dad").

2. **Fact self-containment**: Facts are understandable without the original message. Entity names
   replace pronouns. Specific details (brands, counts, dates, locations) are preserved — not
   generalized ("Gamecube" not "gaming console").

3. **No meta-language**: Summaries and facts state content directly — no "mentioned", "discussed",
   "stated", "noted", or narration of conversational dynamics.

4. **Coverage**: Meaningful preferences, plans, states, and relationships are captured — not
   dropped due to over-conservative extraction. Content-free utterances ("Hi!", "Thanks!") should
   be skipped.

5. **No hallucination**: No attributes, facts, or entities invented beyond what the messages support.
   No reasoning text ("appears to", "(implied by...)") in output fields.

6. **Graph connectivity**: Entities have connecting facts where the messages support them.
   Orphaned entities with no facts are a quality defect when facts were available in the text.
"""


def query_expansion(context: dict[str, Any]) -> list[Message]:
    # Deprecated: not used in production.
    sys_prompt = (
        'You rephrase questions into retrieval-optimized queries for a knowledge graph search system. '
        'Preserve the semantic intent and key entities from the original question.'
    )

    user_prompt = f"""
Rephrase the QUESTION into a simpler third-person query suitable for database retrieval.
Keep relevant context, named entities, and the core information need.

<QUESTION>
{to_prompt_json(context['query'])}
</QUESTION>
"""
    return [
        Message(role='system', content=sys_prompt),
        Message(role='user', content=user_prompt),
    ]


def qa_prompt(context: dict[str, Any]) -> list[Message]:
    # Deprecated: not used in production.
    sys_prompt = (
        'You answer questions from the first-person perspective of the conversation subject. '
        'Use ONLY the provided entity summaries and facts. NEVER invent information beyond them.'
    )

    user_prompt = f"""
Answer the QUESTION briefly as the subject would, using only the ENTITY_SUMMARIES and FACTS below.
If the provided context does not support an answer, say you do not have that information.

<ENTITY_SUMMARIES>
{to_prompt_json(context['entity_summaries'])}
</ENTITY_SUMMARIES>
<FACTS>
{to_prompt_json(context['facts'])}
</FACTS>
<QUESTION>
{context['query']}
</QUESTION>
"""
    return [
        Message(role='system', content=sys_prompt),
        Message(role='user', content=user_prompt),
    ]


def eval_prompt(context: dict[str, Any]) -> list[Message]:
    # Deprecated: not used in production.
    sys_prompt = (
        'You judge whether a RESPONSE correctly answers a QUESTION against a gold-standard ANSWER. '
        'Mark correct when the RESPONSE references the same topic and key facts, even if more verbose.'
    )

    user_prompt = f"""
Given the QUESTION and gold-standard ANSWER, determine if the RESPONSE is correct or incorrect.
Mark correct if the RESPONSE covers the same topic and materially relevant facts as the ANSWER,
even when phrased differently or more verbose. Include reasoning for your grade.

<QUESTION>
{context['query']}
</QUESTION>
<ANSWER>
{context['answer']}
</ANSWER>
<RESPONSE>
{context['response']}
</RESPONSE>
"""
    return [
        Message(role='system', content=sys_prompt),
        Message(role='user', content=user_prompt),
    ]


def eval_add_episode_results(context: dict[str, Any]) -> list[Message]:
    sys_prompt = (
        'You judge graph extraction quality from conversational messages. '
        'Compare a BASELINE extraction against a CANDIDATE extraction for the same MESSAGE.'
    )

    user_prompt = f"""
Given PREVIOUS_MESSAGES and MESSAGE, determine whether the BASELINE graph extraction is higher
quality than the CANDIDATE extraction.

Return candidate_is_worse=False if BASELINE is better.
Return candidate_is_worse=True if CANDIDATE is better or if both are nearly identical in quality.
Add your reasoning to the reasoning field.

{_GRAPH_QUALITY_RUBRIC}

<EXAMPLE>
MESSAGE: "Nisha: My dad is visiting next week. He loves walking his dogs in Riverside Park."
BASELINE entities: ["Nisha", "dad", "dogs", "Riverside Park"] — facts sparse, bare generic nouns.
CANDIDATE entities: ["Nisha", "Nisha's dad", "Riverside Park"] with facts:
  Nisha's dad -> VISITING -> Nisha; Nisha's dad -> WALKS_IN -> Riverside Park.
Result: candidate_is_worse=False (BASELINE has generic entities and weaker coverage).
</EXAMPLE>

<EXAMPLE>
MESSAGE: "Nate: I mostly play on a Gamecube. Last week the windshield on my Mustang got cracked."
BASELINE fact: "Nate plays games" (generalized, lost Gamecube and Mustang details).
CANDIDATE facts: "Nate plays games on a Gamecube"; "The windshield on Nate's Mustang got cracked last week."
Result: candidate_is_worse=True (CANDIDATE preserves specific details and self-contained facts).
</EXAMPLE>

<PREVIOUS_MESSAGES>
{context['previous_messages']}
</PREVIOUS_MESSAGES>
<MESSAGE>
{context['message']}
</MESSAGE>

<BASELINE>
{context['baseline']}
</BASELINE>

<CANDIDATE>
{context['candidate']}
</CANDIDATE>
"""
    return [
        Message(role='system', content=sys_prompt),
        Message(role='user', content=user_prompt),
    ]


versions: Versions = {
    'qa_prompt': qa_prompt,
    'eval_prompt': eval_prompt,
    'query_expansion': query_expansion,
    'eval_add_episode_results': eval_add_episode_results,
}
