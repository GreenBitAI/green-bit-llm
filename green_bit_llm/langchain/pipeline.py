from typing import Any, Iterator, List, Mapping, Optional, Dict
from pydantic import Field
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.llms import BaseLLM
from langchain_core.outputs import Generation, GenerationChunk, LLMResult

from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer, StoppingCriteria, StoppingCriteriaList
import torch
from threading import Thread

from green_bit_llm.common import load
from green_bit_llm.common.utils import check_engine_available

DEFAULT_MODEL_ID = "GreenBitAI/Llama-3-8B-instruct-layer-mix-bpw-4.0"
DEFAULT_TASK = "text-generation"
VALID_TASKS = ("text-generation",)
DEFAULT_BATCH_SIZE = 4


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
            gb = GreenBitPipeline.from_model_id(
                model_id="GreenBitAI/Llama-3-8B-instruct-layer-mix-bpw-4.0",
                task="text-generation",
                pipeline_kwargs={"max_tokens": 100, "temp": 0.7},
                model_kwargs={"dtype": torch.half, "seqlen": 2048}
            )
    """

    pipeline: Any
    model_id: str = DEFAULT_MODEL_ID
    task: str = DEFAULT_TASK
    model_kwargs: Dict[str, Any] = Field(default_factory=dict)
    pipeline_kwargs: Dict[str, Any] = Field(default_factory=dict)
    batch_size: int = DEFAULT_BATCH_SIZE

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model_kwargs = kwargs.get("model_kwargs", {})
        self.pipeline_kwargs = kwargs.get("pipeline_kwargs", {})

    @classmethod
    def from_model_id(
            cls,
            model_id: str,
            task: str = DEFAULT_TASK,
            model_kwargs: Optional[dict] = Field(default_factory=dict),
            pipeline_kwargs: Optional[dict] = Field(default_factory=dict),
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

        model, tokenizer, _ = load(
            model_id,
            **_model_kwargs
        )

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

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
            **kwargs,
        )

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {
            "model_id": self.model_id,
            "task": self.task,
            "model_kwargs": self.model_kwargs,
            "pipeline_kwargs": self.pipeline_kwargs,
        }

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
        pipeline_kwargs = {**self.pipeline_kwargs, **kwargs.get("pipeline_kwargs", {})}
        text_generations = []
        skip_prompt = kwargs.get("skip_prompt", True)

        for i in range(0, len(prompts), self.batch_size):
            batch_prompts = prompts[i: i + self.batch_size]

            # Process batch of prompts
            responses = self.pipeline(
                batch_prompts,
                **pipeline_kwargs,
            )

            # Process each response in the batch
            for j, response in enumerate(responses):
                if isinstance(response, list):
                    # if model returns multiple generations, pick the top one
                    response = response[0]

                if self.pipeline.task == "text-generation":
                    text = response["generated_text"]
                else:
                    raise ValueError(
                        f"Got invalid task {self.pipeline.task}, "
                        f"currently only {VALID_TASKS} are supported"
                    )
                if skip_prompt:
                    text = text[len(batch_prompts[j]):]
                # Append the processed text to results
                text_generations.append(text)

        return LLMResult(
            generations=[[Generation(text=text)] for text in text_generations]
        )

    def _stream(
            self,
            prompt: str,
            stop: Optional[List[str]] = None,
            run_manager: Optional[CallbackManagerForLLMRun] = None,
            **kwargs: Any,
    ) -> Iterator[GenerationChunk]:

        pipeline_kwargs = {**self.pipeline_kwargs, **kwargs.get("pipeline_kwargs", {})}
        skip_prompt = kwargs.get("skip_prompt", True)

        if stop is not None:
            stop_ids = [self.pipeline.tokenizer.encode(s)[-1] for s in stop]
            stopping_criteria = StoppingCriteriaList([StopOnTokens(stop_ids)])
        else:
            stopping_criteria = StoppingCriteriaList()

        streamer = TextIteratorStreamer(
            self.pipeline.tokenizer,
            timeout=60.0,
            skip_prompt=skip_prompt,
            skip_special_tokens=True
        )
        inputs = self.pipeline.tokenizer(prompt, return_tensors="pt").to(self.pipeline.device)

        generation_kwargs = dict(
            **inputs,
            streamer=streamer,
            stopping_criteria=stopping_criteria,
            **pipeline_kwargs,
        )

        thread = Thread(target=self.pipeline.model.generate, kwargs=generation_kwargs)
        thread.start()

        for new_text in streamer:
            chunk = GenerationChunk(text=new_text)
            yield chunk
            if run_manager:
                run_manager.on_llm_new_token(new_text, chunk=chunk)