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
import logging
import openai
import typing
from openai.types.chat import ChatCompletionMessageParam
from pydantic import (
    BaseModel,
    ValidationError,
)

from .config import (
    DEFAULT_MAX_TOKENS,
    ModelSize,
)
from .errors import RateLimitError
from .openai_generic_client import (
    DEFAULT_MODEL,
    OpenAIGenericClient,
)
from ..prompts.models import Message

logger = logging.getLogger(__name__)


class OpenAIJsonObjectClient(OpenAIGenericClient):
    """OpenAI-compatible client that uses ``json_object`` response format.

    Many OpenAI-compatible providers (e.g. Ollama, LM Studio, Qwen via OpenRouter)
    do not support the ``json_schema`` response format. They require:

    1. The word ``json`` to appear in at least one message.
    2. ``response_format`` set to ``json_object`` (not ``json_schema``).

    This client satisfies both requirements by appending the expected JSON schema
    to the system message and requesting ``json_object`` output. Responses are
    validated against the requested Pydantic model; validation failures are reported
    back to the model so it can correct its output (via the retry loop in
    :meth:`OpenAIGenericClient.generate_response`).
    """

    def _schema_instruction_suffix(self, response_model: type[BaseModel] | None) -> str:
        if response_model is not None:
            schema = response_model.model_json_schema()
            return (
                '\n\nRespond with a valid JSON object that strictly follows '
                f'this schema:\n{json.dumps(schema, indent=2)}'
            )
        return '\n\nRespond with a valid JSON object.'

    def _build_openai_messages(
            self,
            messages: list[Message],
            schema_suffix: str,
    ) -> list[ChatCompletionMessageParam]:
        openai_messages: list[ChatCompletionMessageParam] = []
        system_patched = False

        for m in messages:
            content = self._clean_input(m.content)
            if m.role == 'system' and not system_patched:
                content += schema_suffix
                system_patched = True
            if m.role in ('system', 'user', 'assistant'):
                openai_messages.append({'role': m.role, 'content': content})  # type: ignore[misc]

        if not system_patched:
            openai_messages.insert(
                0,
                {'role': 'system', 'content': schema_suffix.strip()},
            )

        return openai_messages

    async def _generate_response(
            self,
            messages: list[Message],
            response_model: type[BaseModel] | None = None,
            max_tokens: int = DEFAULT_MAX_TOKENS,
            model_size: ModelSize = ModelSize.medium,
    ) -> dict[str, typing.Any]:
        schema_suffix = self._schema_instruction_suffix(response_model)
        openai_messages = self._build_openai_messages(messages, schema_suffix)

        try:
            response = await self.client.chat.completions.create(
                model=self.model or DEFAULT_MODEL,
                messages=openai_messages,
                temperature=self.temperature,
                max_tokens=max_tokens or self.max_tokens,
                response_format={'type': 'json_object'},
            )
            result = response.choices[0].message.content or ''
            parsed = json.loads(result)

            if response_model is not None:
                try:
                    return response_model.model_validate(parsed).model_dump()
                except ValidationError:
                    logger.warning(
                        'LLM response failed schema validation: %s',
                        self._get_failed_generation_log(messages, result),
                    )
                    raise

            return parsed
        except openai.RateLimitError as e:
            raise RateLimitError from e
        except Exception as e:
            logger.error(f'Error in generating LLM response: {e}')
            raise
