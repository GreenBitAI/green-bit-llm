from typing import Any, Dict, List, Optional, Union, Sequence, Literal, Callable, Type

from langchain_core.callbacks.manager import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    ChatMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_core.outputs import ChatGeneration, ChatResult, LLMResult
from langchain_core.language_models import LanguageModelInput
from langchain_core.runnables import Runnable, RunnableConfig
from langchain_core.tools import BaseTool
from langchain_core.utils.function_calling import convert_to_openai_tool

from green_bit_llm.langchain import GreenBitPipeline

DEFAULT_SYSTEM_PROMPT = """You are a helpful, respectful, and honest assistant."""


class ChatGreenBit(BaseChatModel):
    """GreenBit Chat model.

    Example:
        .. code-block:: python

            from green_bit_llm.langchain import GreenBitPipeline, ChatGreenBit

            pipeline = GreenBitPipeline.from_model_id(
                model_id="GreenBitAI/Llama-3-8B-instruct-layer-mix-bpw-4.0",
                model_kwargs={"dtype": torch.half, "seqlen": 2048, "requires_grad": False},
                pipeline_kwargs={"max_new_tokens": 100, "temperature": 0.7},
            )

            chat = ChatGreenBit(llm=pipeline)
    """

    llm: GreenBitPipeline
    system_message: SystemMessage = SystemMessage(content=DEFAULT_SYSTEM_PROMPT)

    @property
    def _llm_type(self) -> str:
        return "greenbit-chat"

    def _generate(
            self,
            messages: List[BaseMessage],
            stop: Optional[List[str]] = None,
            run_manager: Optional[CallbackManagerForLLMRun] = None,
            **kwargs: Any,
    ) -> ChatResult:
        llm_input = self._to_chat_prompt(messages)

        # maintain all generation related args
        generation_kwargs = {
            "temperature": kwargs.get("temperature", 0.7),
            "top_p": kwargs.get("top_p", 1.0),
            "repetition_penalty": kwargs.get("repetition_penalty", 1.0),
            "repetition_context_size": kwargs.get("repetition_context_size", 20),
        }

        # Handle max_tokens parameter
        if "max_tokens" in kwargs:
            generation_kwargs["max_new_tokens"] = kwargs["max_tokens"]
        elif "max_new_tokens" in kwargs:
            generation_kwargs["max_new_tokens"] = kwargs["max_new_tokens"]

        if "logit_bias" in kwargs:
            generation_kwargs["logit_bias"] = kwargs["logit_bias"]

        # Wrap generation parameters in pipeline_kwargs
        wrapped_kwargs = {
            "pipeline_kwargs": generation_kwargs,
            "with_hidden_states": kwargs.get("with_hidden_states", False),  # Keep this outside for pipeline's own use
        }

        llm_result = self.llm._generate(
            prompts=[llm_input],
            stop=stop,
            run_manager=run_manager,
            **wrapped_kwargs
        )
        return self._to_chat_result(llm_result)

    def _to_chat_prompt(
            self,
            messages: List[BaseMessage],
    ) -> str:
        """Convert a list of messages into a prompt format expected by wrapped LLM."""
        if not messages:
            raise ValueError("At least one HumanMessage must be provided!")

        if not isinstance(messages[-1], HumanMessage):
            raise ValueError("Last message must be a HumanMessage!")

        messages_dicts = [self._to_chatml_format(m) for m in messages]

        if not hasattr(self.llm, 'pipeline') or not hasattr(self.llm.pipeline, 'tokenizer'):
            raise ValueError("LLM pipeline or tokenizer is not properly initialized")

        return self.llm.pipeline.tokenizer.apply_chat_template(
            messages_dicts, tokenize=False, add_generation_prompt=True
        )

    def _to_chatml_format(self, message: BaseMessage) -> dict:
        """Convert LangChain message to ChatML format."""
        if isinstance(message, SystemMessage):
            role = "system"
        elif isinstance(message, AIMessage):
            role = "assistant"
        elif isinstance(message, HumanMessage):
            role = "user"
        else:
            raise ValueError(f"Unknown message type: {type(message)}")

        return {"role": role, "content": message.content}

    @staticmethod
    def _to_chat_result(llm_result: LLMResult) -> ChatResult:
        """Convert LLM result to chat result, preserving hidden states on their original device."""
        chat_generations = []

        for gen_list in llm_result.generations:
            for g in gen_list:
                generation_info = g.generation_info or {}

                hidden_states = generation_info.get("hidden_states")
                additional_kwargs = {}

                if hidden_states is not None:
                    additional_kwargs["hidden_states"] = hidden_states

                message = AIMessage(
                    content=g.text,
                    additional_kwargs=additional_kwargs
                )

                chat_generation = ChatGeneration(
                    message=message,
                    generation_info=generation_info
                )
                chat_generations.append(chat_generation)

        return ChatResult(
            generations=chat_generations,
            llm_output=llm_result.llm_output
        )

    async def agenerate(
            self,
            messages: List[BaseMessage],
            stop: Optional[List[str]] = None,
            run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
            **kwargs: Any,
    ) -> ChatResult:
        """async generation function"""
        raise NotImplementedError("Async generation not implemented yet")

    def stream(
            self,
            messages: List[BaseMessage],
            stop: Optional[List[str]] = None,
            run_manager: Optional[CallbackManagerForLLMRun] = None,
            **kwargs: Any,
    ):
        """streaming output not support hidden states"""
        prompt = self._to_chat_prompt(messages)

        # Handle generation parameters
        generation_kwargs = {
            "temperature": kwargs.get("temperature", 0.7),
            "top_p": kwargs.get("top_p", 1.0)
        }

        # Handle max_tokens parameter
        if "max_tokens" in kwargs:
            generation_kwargs["max_new_tokens"] = kwargs["max_tokens"]
        elif "max_new_tokens" in kwargs:
            generation_kwargs["max_new_tokens"] = kwargs["max_new_tokens"]

        # Wrap parameters correctly
        wrapped_kwargs = {
            "pipeline_kwargs": generation_kwargs,
            "skip_prompt": kwargs.get("skip_prompt", True)
        }

        for chunk in self.llm._stream(
                prompt,
                stop=stop,
                run_manager=run_manager,
                **wrapped_kwargs
        ):
            yield ChatGeneration(message=AIMessage(content=chunk.text))

    async def astream(
            self,
            messages: List[BaseMessage],
            stop: Optional[List[str]] = None,
            run_manager: Optional[CallbackManagerForLLMRun] = None,
            **kwargs: Any,
    ):
        """async streaming output"""
        raise NotImplementedError("Async stream generation not implemented yet")

    def bind_tools(
            self,
            tools: Sequence[Union[Dict[str, Any], Type, Callable, BaseTool]],
            *,
            tool_choice: Optional[Union[dict, str, Literal["auto", "none"], bool]] = None,
            **kwargs: Any,
    ) -> Runnable[LanguageModelInput, BaseMessage]:
        """Bind tool-like objects to this chat model."""
        formatted_tools = [convert_to_openai_tool(tool) for tool in tools]
        if tool_choice is not None and tool_choice:
            if len(formatted_tools) != 1:
                raise ValueError(
                    "When specifying `tool_choice`, you must provide exactly one "
                    f"tool. Received {len(formatted_tools)} tools."
                )
            if isinstance(tool_choice, str):
                if tool_choice not in ("auto", "none"):
                    tool_choice = {
                        "type": "function",
                        "function": {"name": tool_choice},
                    }
            elif isinstance(tool_choice, bool):
                tool_choice = formatted_tools[0]
            elif isinstance(tool_choice, dict):
                if (
                        formatted_tools[0]["function"]["name"]
                        != tool_choice["function"]["name"]
                ):
                    raise ValueError(
                        f"Tool choice {tool_choice} was specified, but the only "
                        f"provided tool was {formatted_tools[0]['function']['name']}."
                    )
            else:
                raise ValueError(
                    f"Unrecognized tool_choice type. Expected str, bool or dict. "
                    f"Received: {tool_choice}"
                )
            kwargs["tool_choice"] = tool_choice
        return super().bind(tools=formatted_tools, **kwargs)