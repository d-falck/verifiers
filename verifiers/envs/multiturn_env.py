from abc import abstractmethod

import json
from openai import AsyncOpenAI, BadRequestError, APITimeoutError, RateLimitError
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from verifiers.envs.environment import Environment
from verifiers.types import (
    ChatCompletion,
    ChatMessage,
    Completion,
    Info,
    Messages,
    SamplingArgs,
    State,
)
from verifiers.utils.async_utils import maybe_await


class EmptyResponseError(Exception):
    """Raised when the API returns a response with no choices"""
    pass


class MultiTurnEnv(Environment):
    def __init__(self, max_turns: int = -1, inline_reasoning: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.max_turns = max_turns
        self.inline_reasoning = inline_reasoning

    async def setup_state(self, state: State, **kwargs) -> State:
        return state

    @abstractmethod
    async def is_completed(self, messages: Messages, state: State, **kwargs) -> bool:
        pass

    @abstractmethod
    async def env_response(
        self, messages: Messages, state: State, **kwargs
    ) -> tuple[Messages, State]:
        """
        Generate a response from the environment (messages, state).
        """
        pass

    async def rollout(
        self,
        client: AsyncOpenAI,
        model: str,
        prompt: Messages,
        answer: str = "",
        task: str = "default",
        info: Info | None = None,
        sampling_args: SamplingArgs | None = None,
        **kwargs,
    ) -> tuple[Messages, State]:
        """
        Generate a multi-turn rollout with the environment (messages, state).
        """
        info = info or {}
        is_completed = False
        state = {
            "prompt": prompt,
            "completion": [],
            "answer": answer,
            "task": task,
            "info": info,
            "responses": [],
            "turn": 0,
        }
        state = await maybe_await(self.setup_state, state, **kwargs)
        if self.message_type == "chat":
            assert isinstance(prompt, list)
            completion = []
        else:
            assert isinstance(prompt, str)
            completion = ""
            state["responses_start_idx"] = []
        rollout = list(prompt) if not isinstance(prompt, str) else prompt
        while not is_completed:
            try:
                is_completed = await self._single_turn(client, model, info, sampling_args, completion, rollout, state, **kwargs)
            except BadRequestError as e:
                if "context length" in str(e):
                    self.logger.warning(f"Context length exceeded at turn {state['turn']}, truncating rollout: {e}")
                    state["context_truncated"] = True
                    is_completed = True
                else:
                    raise
            # TODO: handle general error.
            # except Exception as e:
            #     self.logger.warning(f"Error getting model response at turn {state['turn']}, skipping rollout: {e}")
            #     state["error"] = str(e)
            #     is_completed = True
        return completion, state

    async def _single_turn(self, client, model, info, sampling_args, completion, rollout, state, **kwargs) -> bool:
        @retry(
            stop=stop_after_attempt(4),
            wait=wait_exponential(multiplier=1, min=2, max=10),
            retry=retry_if_exception_type((EmptyResponseError, json.JSONDecodeError, APITimeoutError, RateLimitError)),
            before_sleep=lambda retry_state: self.logger.warning(
                f"API error ({type(retry_state.outcome.exception()).__name__}), retrying (attempt {retry_state.attempt_number + 1}/4)"
            )
        )
        async def inner():
            nonlocal completion, rollout, state
            
            if await maybe_await(self.is_completed, rollout, state, **kwargs):
                return True
            response = await self.get_model_response(
                client=client,
                model=model,
                prompt=rollout,
                oai_tools=info.get("oai_tools", None),
                sampling_args=sampling_args,
                message_type=self.message_type,
            )

            if not response or not hasattr(response, 'choices') or not response.choices:
                raise EmptyResponseError(f"API returned empty response at turn {state['turn']}")

            state["responses"].append(response)
            if self.message_type == "chat":
                assert isinstance(rollout, list)
                assert isinstance(completion, list)
                assert isinstance(response, ChatCompletion)
                response_text: str = self._extract_response_text(response)
                response_message: ChatMessage = {
                    "role": "assistant",
                    "content": response_text,
                }
                if response.choices[0].message.tool_calls:
                    response_message["tool_calls"] = response.choices[  # type: ignore
                        0
                    ].message.tool_calls
                rollout.append(response_message)
                completion.append(response_message)
            else:
                assert isinstance(rollout, str)
                assert isinstance(completion, str)
                assert isinstance(response, Completion)
                state["responses_start_idx"].append(len(completion))
                response_text: str = response.choices[0].text or ""  # type: ignore
                rollout += response_text
                completion += response_text
            state["turn"] += 1
            if await maybe_await(self.is_completed, rollout, state, **kwargs) or (
                state["turn"] >= self.max_turns and self.max_turns > 0
            ):
                return True

            env_msgs, state = await maybe_await(
                self.env_response, rollout, state, **kwargs
            )
            if self.message_type == "chat":
                assert isinstance(env_msgs, list)
                assert isinstance(rollout, list)
                assert isinstance(completion, list)
                rollout += env_msgs
                completion += env_msgs
            else:
                assert isinstance(env_msgs, str)
                assert isinstance(rollout, str)
                assert isinstance(completion, str)
                rollout += env_msgs
                completion += env_msgs

            return False

        return await inner()

    def _extract_response_text(self, response: ChatCompletion) -> str:
        response_text = response.choices[0].message.content or ""
        if not self.inline_reasoning:
            return response_text

        message = response.choices[0].message
        reasoning_text = None
        if hasattr(message, 'reasoning') and message.reasoning:
            reasoning_text = message.reasoning
        # elif hasattr(message, '__pydantic_extra__') and 'reasoning' in message.__pydantic_extra__:
        #     reasoning_text = message.__pydantic_extra__['reasoning']

        if reasoning_text:
            reasoning_text = reasoning_text.rstrip()
            response_text = f"<think>\n{reasoning_text}\n</think>\n\n{response_text}"
        return response_text