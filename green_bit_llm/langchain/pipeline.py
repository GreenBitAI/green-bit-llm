from typing import Any, Iterator, List, Mapping, Optional, Dict
from pydantic import Field
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.llms import BaseLLM
from langchain_core.outputs import Generation, GenerationChunk, LLMResult

from transformers import (
    pipeline, AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer, StoppingCriteria, StoppingCriteriaList
)
import torch
from threading import Thread

from green_bit_llm.common import load
from green_bit_llm.common.utils import check_engine_available

DEFAULT_MODEL_ID = "GreenBitAI/Llama-3-8B-instruct-layer-mix-bpw-4.0"
DEFAULT_TASK = "text-generation"
VALID_TASKS = ("text-generation",)
DEFAULT_BATCH_SIZE = 1


class StopOnTokens(StoppingCriteria):
    def __init__(self, stop_ids: List[int]):
        self.stop_ids = stop_ids

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        for stop_id in self.stop_ids:
            if input_ids[0][-1] == stop_id:
                return True
        return False


class GreenBitPipeline(BaseLLM):
    """GreenBit Pipeline API.

    To use, you should have the BitorchEngine and transformers library installed.

    Only supports `text-generation` for now.

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

    pipeline: Any = Field(default=None)
    model_id: str = Field(default=DEFAULT_MODEL_ID)
    task: str = Field(default=DEFAULT_TASK)
    model_kwargs: Dict[str, Any] = Field(default_factory=dict)
    pipeline_kwargs: Dict[str, Any] = Field(default_factory=dict)
    batch_size: int = Field(default=DEFAULT_BATCH_SIZE)

    class Config:
        """Configuration for this pydantic object."""
        arbitrary_types_allowed = True

    @classmethod
    def from_model_id(
            cls,
            model_id: str,
            task: str = DEFAULT_TASK,
            model_kwargs: Optional[Dict[str, Any]] = Field(default_factory=dict),
            pipeline_kwargs: Optional[Dict[str, Any]] = Field(default_factory=dict),
            batch_size: int = DEFAULT_BATCH_SIZE,
            **kwargs: Any,
    ) -> "GreenBitPipeline":
        if not check_engine_available():
            raise ValueError(
                "Could not import BitorchEngine. "
                "Please ensure it is properly installed."
            )

        if task not in VALID_TASKS:
            raise ValueError(
                f"Got invalid task {task}, "
                f"currently only {VALID_TASKS} are supported"
            )

        _model_kwargs = model_kwargs or {"dtype": torch.half, "seqlen": 2048, "device_map": "auto"}
        _pipeline_kwargs = pipeline_kwargs or {"max_tokens": 100, "temp": 0.7}

        model, tokenizer, _ = load(model_id, **_model_kwargs)

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id

        pipe = pipeline(
            task=task,
            model=model,
            tokenizer=tokenizer,
            **_pipeline_kwargs
        )

        return cls(
            pipeline=pipe,
            model_id=model_id,
            task=task,
            model_kwargs=_model_kwargs,
            pipeline_kwargs=_pipeline_kwargs,
            batch_size=batch_size
        )

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {
            "model_id": self.model_id,
            "task": self.task,
            "model_kwargs": self.model_kwargs,
            "pipeline_kwargs": self.pipeline_kwargs,
            "batch_size": self.batch_size
        }

    def _prepare_generation_config(self, pipeline_kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """prepare generation configuration"""
        merged_kwargs = {**self.pipeline_kwargs, **pipeline_kwargs}

        generation_config = {
            "do_sample": merged_kwargs.get("do_sample", False),
            "pad_token_id": self.pipeline.tokenizer.pad_token_id,
            "eos_token_id": self.pipeline.tokenizer.eos_token_id,
            "temperature": merged_kwargs.get("temperature", 0.7),
            "top_p": merged_kwargs.get("top_p", 0.95),
            "top_k": merged_kwargs.get("top_k", 50),
            "max_new_tokens": merged_kwargs.get("max_new_tokens", 100),
            "repetition_penalty": merged_kwargs.get("repetition_penalty", 1.1),
            "num_return_sequences": 1,
        }

        return generation_config

    def _prepare_prompt_from_text(self, text: str, **kwargs) -> str:
        """Convert plain text to chat format and apply template"""
        if not hasattr(self.pipeline.tokenizer, 'apply_chat_template'):
            # Fallback: return text as-is if no chat template support
            return text

        try:
            # Convert text to chat message format
            messages = [{"role": "user", "content": text}]

            # Prepare template arguments
            template_kwargs = {
                "add_generation_prompt": True,
                "tokenize": False,  # 添加这行，确保返回字符串
            }

            # Add enable_thinking for Qwen3 models if provided
            enable_thinking = kwargs.get('enable_thinking')
            if enable_thinking is not None:
                template_kwargs["enable_thinking"] = enable_thinking

            result = self.pipeline.tokenizer.apply_chat_template(messages, **template_kwargs)

            # 确保返回字符串
            if isinstance(result, list):
                return self.pipeline.tokenizer.decode(result, skip_special_tokens=False)
            return result

        except Exception:
            # If template application fails, return original text
            return text

    @property
    def _llm_type(self) -> str:
        return "greenbit_pipeline"

    def generate(
            self,
            prompts: List[str],
            stop: Optional[List[str]] = None,
            run_manager: Optional[CallbackManagerForLLMRun] = None,
            **kwargs: Any,
    ) -> LLMResult:
        # Process prompts through chat template if they're plain text
        processed_prompts = []
        for prompt in prompts:
            # If prompt doesn't look like it's already formatted, apply chat template
            if not any(marker in prompt for marker in
                       ['<|im_start|>', '<|start_header_id|>', '[INST]', '<start_of_turn>']):
                processed_prompts.append(self._prepare_prompt_from_text(prompt, **kwargs))
            else:
                processed_prompts.append(prompt)

        # Get and merge pipeline kwargs
        pipeline_kwargs = {**self.pipeline_kwargs}
        if "pipeline_kwargs" in kwargs:
            pipeline_kwargs.update(kwargs["pipeline_kwargs"])

        generation_config = self._prepare_generation_config(pipeline_kwargs)

        if stop:
            stop_ids = [self.pipeline.tokenizer.encode(s)[-1] for s in stop]
            generation_config["stopping_criteria"] = StoppingCriteriaList([
                StopOnTokens(stop_ids)
            ])

        text_generations = []
        hidden_states_list = []
        with_hidden_states = kwargs.get("with_hidden_states", False)

        for i in range(0, len(processed_prompts), self.batch_size):
            batch_prompts = processed_prompts[i: i + self.batch_size]

            inputs = self.pipeline.tokenizer(
                batch_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True
            ).to(self.pipeline.device)

            input_lengths = [len(ids) for ids in inputs['input_ids']]

            # modifies pipeline kwargs for getting hidden states
            if with_hidden_states:
                generation_config["output_hidden_states"] = True

            outputs = self.pipeline.model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                **generation_config,
                return_dict_in_generate=True,
                output_scores=True,
            )

            # processing hidden states
            if with_hidden_states:
                hidden_layer = -1
                # get hidden states of inputs
                input_tokens_hs = outputs.hidden_states[0][hidden_layer]
                # use attention mask
                mask = inputs["attention_mask"].unsqueeze(2)
                # calculates mean
                batch_embeddings = torch.sum(input_tokens_hs * mask, dim=1) / torch.sum(mask, dim=1)
                # keep batch dimension when adding to list
                hidden_states_list.extend([emb.unsqueeze(0).detach() for emb in batch_embeddings])

            generated_texts = []
            for j, sequence in enumerate(outputs.sequences):
                response_tokens = sequence[input_lengths[j]:]
                decoded_text = self.pipeline.tokenizer.decode(
                    response_tokens,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True
                )
                generated_texts.append(decoded_text)

            # For chat templates, the decoded text is already clean
            for text in generated_texts:
                text_generations.append(text.strip())

        generations = []
        for i, text in enumerate(text_generations):
            generation_info = {}
            if with_hidden_states and i < len(hidden_states_list):
                generation_info["hidden_states"] = hidden_states_list[i]

            generations.append([Generation(text=text, generation_info=generation_info)])

        return LLMResult(generations=generations)

    def _generate(
            self,
            prompts: List[str],
            stop: Optional[List[str]] = None,
            run_manager: Optional[CallbackManagerForLLMRun] = None,
            **kwargs: Any,
    ) -> LLMResult:
        """Generate method required by BaseLLM"""
        return self.generate(prompts, stop, run_manager, **kwargs)

    def stream(
            self,
            prompt: str,
            stop: Optional[List[str]] = None,
            run_manager: Optional[CallbackManagerForLLMRun] = None,
            **kwargs: Any,
    ) -> Iterator[GenerationChunk]:
        # Process prompt through chat template if it's plain text
        if not any(marker in prompt for marker in ['<|im_start|>', '<|start_header_id|>', '[INST]', '<start_of_turn>']):
            processed_prompt = self._prepare_prompt_from_text(prompt, **kwargs)
        else:
            processed_prompt = prompt

        # Get and merge pipeline kwargs
        pipeline_kwargs = {**self.pipeline_kwargs}
        if "pipeline_kwargs" in kwargs:
            pipeline_kwargs.update(kwargs["pipeline_kwargs"])

        generation_config = self._prepare_generation_config(pipeline_kwargs)

        streamer = TextIteratorStreamer(
            self.pipeline.tokenizer,
            timeout=60.0,
            skip_prompt=kwargs.get("skip_prompt", True),
            skip_special_tokens=True
        )
        inputs = self.pipeline.tokenizer(processed_prompt, return_tensors="pt").to(self.pipeline.device)

        generation_kwargs = dict(
            **inputs,
            streamer=streamer,
            **generation_config,
        )

        thread = Thread(target=self.pipeline.model.generate, kwargs=generation_kwargs)
        thread.start()

        for new_text in streamer:
            chunk = GenerationChunk(text=new_text)
            yield chunk
            if run_manager:
                run_manager.on_llm_new_token(new_text, chunk=chunk)