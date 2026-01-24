from abc import abstractmethod

import json
from openai import AsyncOpenAI, BadRequestError, APITimeoutError, RateLimitError, APIStatusError
from tenacity import retry, stop_after_attempt, wait_random_exponential, retry_if_exception_type

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


API_NUM_RETRIES = 10


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
            stop=stop_after_attempt(API_NUM_RETRIES),
            wait=wait_random_exponential(multiplier=1, min=2, max=10),
            retry=retry_if_exception_type((EmptyResponseError, json.JSONDecodeError, APITimeoutError, RateLimitError, APIStatusError)),
            before_sleep=lambda retry_state: self.logger.warning(
                f"API error ({type(retry_state.outcome.exception()).__name__}), retrying (attempt {retry_state.attempt_number + 1}/{API_NUM_RETRIES})"
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

        try:
            return await inner()
        except Exception as e:
            # All retries failed - log error and skip this rollout instead of crashing
            self.logger.error(
                f"Failed to complete rollout after {API_NUM_RETRIES} attempts: {type(e).__name__}: {e}"
            )
            self.logger.error(f"Skipping rollout to continue evaluation")
            # Mark the state as failed so we can track which rollouts failed
            state["api_error"] = {
                "type": type(e).__name__,
                "message": str(e),
                "turn": state["turn"],
            }
            # Return True to indicate this rollout is "complete" (even though it failed)
            # This allows the evaluation to continue instead of crashing
            return True

    def _extract_response_text(self, response: ChatCompletion) -> str:
        response_text = response.choices[0].message.content or ""
        if not self.inline_reasoning:
            return response_text

        message = response.choices[0].message

        # DEBUG: Log message attributes to see what the API returns
        # self.logger.info(f"[DEBUG REASONING] Message attributes: {dir(message)}")
        # self.logger.info(f"[DEBUG REASONING] Message dict: {message.model_dump() if hasattr(message, 'model_dump') else 'N/A'}")

        reasoning_text = None

        # Check for reasoning_details (OpenRouter format - array of objects with 'text' field)
        if hasattr(message, 'reasoning_details') and message.reasoning_details:
            # self.logger.info(f"[DEBUG REASONING] Found reasoning_details array: {len(message.reasoning_details)} items")
            # Extract text from reasoning_details array
            reasoning_parts = []
            for detail in message.reasoning_details:
                if hasattr(detail, 'text') and detail.text:
                    reasoning_parts.append(detail.text)
                elif isinstance(detail, dict) and 'text' in detail:
                    reasoning_parts.append(detail['text'])
            if reasoning_parts:
                reasoning_text = '\n'.join(reasoning_parts)
                # self.logger.info(f"[DEBUG REASONING] Extracted reasoning from reasoning_details: {reasoning_text[:200]}...")
        # Check for reasoning_content (older OpenRouter format)
        elif hasattr(message, 'reasoning_content') and message.reasoning_content:
            # self.logger.info(f"[DEBUG REASONING] Found reasoning_content: {message.reasoning_content[:200]}...")
            reasoning_text = message.reasoning_content
        # Check for reasoning (DeepSeek, o1, etc.)
        elif hasattr(message, 'reasoning') and message.reasoning:
            # self.logger.info(f"[DEBUG REASONING] Found reasoning: {message.reasoning[:200]}...")
            reasoning_text = message.reasoning
        # Check __pydantic_extra__ for non-standard fields
        elif hasattr(message, '__pydantic_extra__') and message.__pydantic_extra__:
            # self.logger.info(f"[DEBUG REASONING] Checking __pydantic_extra__: {message.__pydantic_extra__}")
            if 'reasoning_details' in message.__pydantic_extra__:
                reasoning_details = message.__pydantic_extra__['reasoning_details']
                # self.logger.info(f"[DEBUG REASONING] Found reasoning_details in __pydantic_extra__: {reasoning_details}")
                if isinstance(reasoning_details, list):
                    reasoning_parts = []
                    for detail in reasoning_details:
                        if isinstance(detail, dict) and 'text' in detail:
                            reasoning_parts.append(detail['text'])
                    if reasoning_parts:
                        reasoning_text = '\n'.join(reasoning_parts)
            elif 'reasoning' in message.__pydantic_extra__:
                reasoning_text = message.__pydantic_extra__['reasoning']
                # self.logger.info(f"[DEBUG REASONING] Found reasoning in __pydantic_extra__")

        if not reasoning_text:
            # self.logger.info(f"[DEBUG REASONING] No reasoning found. Has reasoning_details: {hasattr(message, 'reasoning_details')}, Has reasoning_content: {hasattr(message, 'reasoning_content')}, Has reasoning: {hasattr(message, 'reasoning')}")
            pass

        if reasoning_text:
            reasoning_text = reasoning_text.rstrip()
            response_text = f"<think>\n{reasoning_text}\n</think>\n\n{response_text}"
            # self.logger.info(f"[DEBUG REASONING] Final response_text length: {len(response_text)}, has <think> tags: {response_text.startswith('<think>')}")
        else:
            # self.logger.info(f"[DEBUG REASONING] No reasoning_text to prepend")
            pass

        return response_text