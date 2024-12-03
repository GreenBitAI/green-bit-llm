from typing import Any, Iterator, List, Mapping, Optional, Dict
from pydantic import Field
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.llms import BaseLLM
from langchain_core.outputs import Generation, GenerationChunk, LLMResult

from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer, StoppingCriteria, StoppingCriteriaList
import torch
from threading import Thread
import re

from green_bit_llm.common import load
from green_bit_llm.common.utils import check_engine_available
from green_bit_llm.inference.conversation import get_conv_template, SeparatorStyle, Conversation
from .response_format_handler import ResponseFormatHandler

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
    conv_template: Optional[Conversation] = Field(default=None)
    response_format_handler: Optional[ResponseFormatHandler] = Field(default=None)

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

        conv_template = cls._get_conversation_template(cls, model_id)

        model, tokenizer, _ = load(
            model_id,
            **_model_kwargs
        )

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
            batch_size=batch_size,
            conv_template=conv_template,
            response_format_handler=ResponseFormatHandler(conv_template)
        )

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {
            "model_id": self.model_id,
            "task": self.task,
            "model_kwargs": self.model_kwargs,
            "pipeline_kwargs": self.pipeline_kwargs,
            "batch_size": self.batch_size,
            "conv_template": self.conv_template
        }

    def _get_conversation_template(self, model_id) -> Optional[Conversation]:
        """Get appropriate conversation template based on model ID"""
        model_id_lower = model_id.lower()

        if "qwen" in model_id_lower:
            return get_conv_template("qwen-chat")
        elif "llama-3" in model_id_lower:
            return get_conv_template("llama-3")
        elif "llama-2" in model_id_lower:
            return get_conv_template("llama-2")
        elif "mistral" in model_id_lower:
            return get_conv_template("mistral")
        elif "yi" in model_id_lower:
            return get_conv_template("yi-chat")
        elif "phi-3" in model_id_lower:
            return get_conv_template("phi-3")
        elif "gemma" in model_id_lower:
            return get_conv_template("gemma")
        elif "tinyllama" in model_id_lower:
            return get_conv_template("TinyLlama")
        elif "gemini" in model_id_lower:
            return get_conv_template("gemini")

        return None

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

        if self.conv_template:
            if self.conv_template.stop_token_ids:
                generation_config["stopping_criteria"] = StoppingCriteriaList([
                    StopOnTokens(self.conv_template.stop_token_ids)
                ])
            if self.conv_template.stop_str:
                generation_config["stopping_criteria"] = StoppingCriteriaList([
                    StopOnTokens([
                        self.pipeline.tokenizer.encode(self.conv_template.stop_str)[-1]
                    ])
                ])

        return generation_config

    @property
    def _llm_type(self) -> str:
        return "greenbit_pipeline"


    def _generate(
            self,
            prompts: List[str],
            stop: Optional[List[str]] = None,
            run_manager: Optional[CallbackManagerForLLMRun] = None,
            **kwargs: Any,
    ) -> LLMResult:
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

        for i in range(0, len(prompts), self.batch_size):
            batch_prompts = prompts[i: i + self.batch_size]

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
                # input_tokens_hs = input_tokens_hs * mask
                # calculates mean
                batch_embeddings = torch.sum(input_tokens_hs*mask, dim=1) / torch.sum(mask, dim=1)
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

            for j, text in enumerate(generated_texts):
                if kwargs.get("skip_prompt", True):
                    if self.conv_template:
                        text = self.response_format_handler.extract_response(text, clean_template=True)
                    else:
                        text = text[len(batch_prompts[j]):]
                text_generations.append(text.strip())

        generations = []
        for i, text in enumerate(text_generations):
            generation_info = {}
            if with_hidden_states and i < len(hidden_states_list):
                generation_info["hidden_states"] = hidden_states_list[i]

            generations.append([Generation(text=text, generation_info=generation_info)])

        return LLMResult(generations=generations)


    def _stream(
            self,
            prompt: str,
            stop: Optional[List[str]] = None,
            run_manager: Optional[CallbackManagerForLLMRun] = None,
            **kwargs: Any,
    ) -> Iterator[GenerationChunk]:

        # Get and merge pipeline kwargs
        pipeline_kwargs = {**self.pipeline_kwargs}
        if "pipeline_kwargs" in kwargs:
            pipeline_kwargs.update(kwargs["pipeline_kwargs"])

        generation_config = self._prepare_generation_config(pipeline_kwargs)

        if self.conv_template:
            conv = self.conv_template.copy()
            conv.append_message(conv.roles[0], prompt)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()


        streamer = TextIteratorStreamer(
            self.pipeline.tokenizer,
            timeout=60.0,
            skip_prompt=kwargs.get("skip_prompt", True),
            skip_special_tokens=True
        )
        inputs = self.pipeline.tokenizer(prompt, return_tensors="pt").to(self.pipeline.device)

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