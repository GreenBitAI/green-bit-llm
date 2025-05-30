from typing import Any, Dict, List, Optional, Union, Sequence, Literal, Callable, Type
from pydantic import Field
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

            from green_bit_llm.langchain import GreenBitPipeline

            model_config = {
                "trust_remote_code": True,
                "attn_implementation": "flash_attention_2"
            }

            tokenizer_config = {"trust_remote_code": True}

            gb = GreenBitPipeline.from_model_id(
                model_id="GreenBitAI/Llama-3-8B-instruct-layer-mix-bpw-4.0",
                task="text-generation",
                pipeline_kwargs={"max_tokens": 100, "temp": 0.7},
                model_kwargs={
                    "dtype": torch.half,
                    "seqlen": 2048,
                    "requires_grad": False,
                    "model_config": model_config,
                    "tokenizer_config": tokenizer_config
                }
            )
    """

    llm: GreenBitPipeline = Field(..., description="GreenBit Pipeline instance")

    class Config:
        """Configuration for this pydantic object."""
        arbitrary_types_allowed = True

    def __init__(
            self,
            llm: GreenBitPipeline,
            **kwargs: Any,
    ) -> None:
        """Initialize the chat model.

        Args:
            llm: GreenBit Pipeline instance
            **kwargs: Additional keyword arguments
        """
        # First initialize with mandatory llm field
        super().__init__(llm=llm, **kwargs)

    @property
    def _llm_type(self) -> str:
        return "greenbit-chat"

    def _create_chat_result(self, llm_result: LLMResult) -> ChatResult:
        """Convert LLM result to chat messages"""
        generations = []
        for gen in llm_result.generations:
            for g in gen:
                message = AIMessage(content=g.text.strip())
                chat_generation = ChatGeneration(
                    message=message,
                    generation_info=g.generation_info
                )
                generations.append(chat_generation)
        return ChatResult(generations=generations)

    def _messages_to_dict(self, messages: List[BaseMessage]) -> List[Dict[str, str]]:
        """Convert LangChain messages to dictionary format for apply_chat_template"""
        message_dicts = []

        for message in messages:
            if isinstance(message, SystemMessage):
                message_dicts.append({"role": "system", "content": message.content})
            elif isinstance(message, HumanMessage):
                message_dicts.append({"role": "user", "content": message.content})
            elif isinstance(message, AIMessage):
                message_dicts.append({"role": "assistant", "content": message.content})

        return message_dicts

    def _prepare_prompt(self, messages: List[BaseMessage], **kwargs) -> str:
        """Prepare prompt using apply_chat_template"""
        if not hasattr(self.llm.pipeline, 'tokenizer'):
            raise ValueError("Tokenizer not available in pipeline")

        tokenizer = self.llm.pipeline.tokenizer
        if not hasattr(tokenizer, 'apply_chat_template'):
            raise ValueError("Tokenizer does not support apply_chat_template")

        # Convert messages to dict format
        message_dicts = self._messages_to_dict(messages)

        # Prepare apply_chat_template arguments
        template_kwargs = {
            "add_generation_prompt": True,
            "tokenize": False,  # 关键：设置为 False 确保返回字符串
        }

        # Add enable_thinking for Qwen3 models if provided
        enable_thinking = kwargs.get('enable_thinking')
        if enable_thinking is not None:
            template_kwargs["enable_thinking"] = enable_thinking

        try:
            result = tokenizer.apply_chat_template(message_dicts, **template_kwargs)

            # 确保返回字符串
            if isinstance(result, list):
                # 如果仍然返回 token IDs，解码为字符串
                return tokenizer.decode(result, skip_special_tokens=False)
            elif isinstance(result, str):
                return result
            else:
                raise ValueError(f"Unexpected return type from apply_chat_template: {type(result)}")

        except Exception as e:
            raise ValueError(f"Failed to apply chat template: {str(e)}")

    def generate(
            self,
            messages: List[BaseMessage],
            stop: Optional[List[str]] = None,
            run_manager: Optional[CallbackManagerForLLMRun] = None,
            **kwargs: Any,
    ) -> ChatResult:
        """Generate chat completion using the underlying pipeline"""
        # Prepare prompt using apply_chat_template
        prompt = self._prepare_prompt(messages, **kwargs)

        # Handle generation parameters
        generation_kwargs = {}
        if "temperature" in kwargs:
            generation_kwargs["temperature"] = kwargs["temperature"]
        if "max_new_tokens" in kwargs:
            generation_kwargs["max_new_tokens"] = kwargs["max_new_tokens"]
        elif "max_tokens" in kwargs:
            generation_kwargs["max_new_tokens"] = kwargs["max_tokens"]

        wrapped_kwargs = {
            "pipeline_kwargs": generation_kwargs,
            "stop": stop,
            **kwargs
        }

        # Generate using pipeline
        llm_result = self.llm.generate(
            prompts=[prompt],
            run_manager=run_manager,
            **wrapped_kwargs
        )

        return self._create_chat_result(llm_result)

    def _generate(
            self,
            messages: List[BaseMessage],
            stop: Optional[List[str]] = None,
            run_manager: Optional[CallbackManagerForLLMRun] = None,
            **kwargs: Any,
    ) -> ChatResult:
        """Generate method required by BaseChatModel"""
        return self.generate(messages, stop, run_manager, **kwargs)

    def stream(
            self,
            messages: List[BaseMessage],
            stop: Optional[List[str]] = None,
            run_manager: Optional[CallbackManagerForLLMRun] = None,
            **kwargs: Any,
    ):
        """Stream chat completion"""
        # Prepare prompt using apply_chat_template
        prompt = self._prepare_prompt(messages, **kwargs)

        # Handle generation parameters
        generation_kwargs = {}
        if "temperature" in kwargs:
            generation_kwargs["temperature"] = kwargs["temperature"]
        if "max_new_tokens" in kwargs:
            generation_kwargs["max_new_tokens"] = kwargs["max_new_tokens"]
        elif "max_tokens" in kwargs:
            generation_kwargs["max_new_tokens"] = kwargs["max_tokens"]

        wrapped_kwargs = {
            "pipeline_kwargs": generation_kwargs,
            "stop": stop,
            "skip_prompt": kwargs.get("skip_prompt", True),
            **kwargs
        }

        # Stream using pipeline
        for chunk in self.llm.stream(
                prompt,
                run_manager=run_manager,
                **wrapped_kwargs
        ):
            yield ChatGeneration(message=AIMessage(content=chunk.text))

    async def agenerate(
            self,
            messages: List[BaseMessage],
            stop: Optional[List[str]] = None,
            run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
            **kwargs: Any,
    ) -> ChatResult:
        """Async generation (not implemented)"""
        raise NotImplementedError("Async generation not implemented yet")

    async def astream(
            self,
            messages: List[BaseMessage],
            stop: Optional[List[str]] = None,
            run_manager: Optional[CallbackManagerForLLMRun] = None,
            **kwargs: Any,
    ):
        """Async streaming (not implemented)"""
        raise NotImplementedError("Async stream generation not implemented yet")

    def bind_tools(
            self,
            tools: Sequence[Union[Dict[str, Any], Type, Callable, BaseTool]],
            *,
            tool_choice: Optional[Union[dict, str, Literal["auto", "none"], bool]] = None,
            **kwargs: Any,
    ) -> Runnable[LanguageModelInput, BaseMessage]:
        """Bind tools to the chat model"""
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