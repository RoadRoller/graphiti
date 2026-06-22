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

import pytest
from pydantic import (
    BaseModel,
    ValidationError,
)
from unittest.mock import (
    AsyncMock,
    MagicMock,
)

from graphiti_core.llm_client.config import LLMConfig
from graphiti_core.llm_client.openai_json_object_client import OpenAIJsonObjectClient
from graphiti_core.prompts.models import Message


class SampleResponse(BaseModel):
    name: str
    count: int = 0


@pytest.fixture
def mock_async_openai():
    mock_client = MagicMock()
    mock_client.chat.completions.create = AsyncMock()
    return mock_client


@pytest.fixture
def json_object_client(mock_async_openai):
    config = LLMConfig(
        api_key='test-key',
        model='test-model',
        temperature=0.0,
        max_tokens=1024,
    )
    client = OpenAIJsonObjectClient(config=config, client=mock_async_openai)
    return client


def _mock_completion(content: str) -> MagicMock:
    message = MagicMock()
    message.content = content
    choice = MagicMock()
    choice.message = message
    response = MagicMock()
    response.choices = [choice]
    return response


@pytest.mark.asyncio
async def test_uses_json_object_response_format(json_object_client, mock_async_openai):
    mock_async_openai.chat.completions.create.return_value = _mock_completion('{"name": "alice"}')
    messages = [
        Message(role='system', content='Extract entities.'),
        Message(role='user', content='Alice went to the store.'),
    ]

    result = await json_object_client.generate_response(messages, response_model=SampleResponse)

    assert result == {'name': 'alice', 'count': 0}
    call_kwargs = mock_async_openai.chat.completions.create.call_args.kwargs
    assert call_kwargs['response_format'] == {'type': 'json_object'}
    sent_messages = call_kwargs['messages']
    assert 'json' in sent_messages[0]['content'].lower()
    assert 'SampleResponse' in sent_messages[0]['content'] or 'name' in sent_messages[0]['content']


@pytest.mark.asyncio
async def test_injects_system_message_when_missing(json_object_client, mock_async_openai):
    mock_async_openai.chat.completions.create.return_value = _mock_completion('{"key": "v"}')
    messages = [Message(role='user', content='hello')]

    await json_object_client.generate_response(messages)

    call_kwargs = mock_async_openai.chat.completions.create.call_args.kwargs
    assert call_kwargs['messages'][0]['role'] == 'system'
    assert 'json' in call_kwargs['messages'][0]['content'].lower()


@pytest.mark.asyncio
async def test_validation_error_triggers_retry(json_object_client, mock_async_openai):
    mock_async_openai.chat.completions.create.side_effect = [
        _mock_completion('{"wrong_field": "x"}'),
        _mock_completion('{"name": "fixed"}'),
    ]
    messages = [Message(role='user', content='test')]

    result = await json_object_client.generate_response(messages, response_model=SampleResponse)

    assert mock_async_openai.chat.completions.create.call_count == 2
    assert result == {'name': 'fixed', 'count': 0}
    assert any('invalid' in m.content.lower() for m in messages if m.role == 'user')


@pytest.mark.asyncio
async def test_validation_error_exhausts_retries(json_object_client, mock_async_openai):
    mock_async_openai.chat.completions.create.return_value = _mock_completion(
        '{"wrong_field": "x"}'
    )
    messages = [Message(role='user', content='test')]

    with pytest.raises(ValidationError):
        await json_object_client.generate_response(messages, response_model=SampleResponse)

    assert (
            mock_async_openai.chat.completions.create.call_count
            == OpenAIJsonObjectClient.MAX_RETRIES + 1
    )
