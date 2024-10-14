from __future__ import annotations

from typing import Any, Iterator, List, Mapping, Optional

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
                model_kwargs={"dtype": torch.half, "device_map": 'auto', "seqlen": 2048, "requires_grad": False}
            )
    """

    pipeline: Any
    model_id: str = DEFAULT_MODEL_ID
    task: str = DEFAULT_TASK
    model_kwargs: Optional[dict] = None
    pipeline_kwargs: Optional[dict] = None
    batch_size: int = DEFAULT_BATCH_SIZE

    class Config:
        """Configuration for this pydantic object."""
        extra = "forbid"

    @classmethod
    def from_model_id(
        cls,
        model_id: str,
        task: str = DEFAULT_TASK,
        device: Optional[str] = "cuda:0",
        model_kwargs: Optional[dict] = None,
        pipeline_kwargs: Optional[dict] = None,
        batch_size: int = DEFAULT_BATCH_SIZE,
        **kwargs: Any,
    ) -> GreenBitPipeline:
        """Construct the pipeline object from model_id and task."""
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

        _model_kwargs = model_kwargs or {}
        _pipeline_kwargs = pipeline_kwargs or {}

        # Load model and tokenizer
        model, tokenizer, _ = load(
            model_id,
            device_map={device},
            **_model_kwargs
        )

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Create pipeline
        pipe = pipeline(
            task=task,
            model=model,
            tokenizer=tokenizer,
            device=device,
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
        """Get the identifying parameters."""
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
        # List to hold all results
        text_generations: List[str] = []
        pipeline_kwargs = {**self.pipeline_kwargs, **kwargs.get("pipeline_kwargs", {})}

        for i in range(0, len(prompts), self.batch_size):
            batch_prompts = prompts[i : i + self.batch_size]

            # Process batch of prompts
            responses = self.pipeline(batch_prompts, **pipeline_kwargs)

            # Process each response in the batch
            for response in responses:
                if isinstance(response, list):
                    # if model returns multiple generations, pick the top one
                    response = response[0]

                if self.pipeline.task == "text-generation":
                    text = response["generated_text"]
                else:
                    raise ValueError(f"Unsupported task: {self.pipeline.task}")

                # Remove the prompt from the generated text
                text = text[len(batch_prompts[0]):]
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

        # Create stopping criteria
        if stop is not None:
            stop_ids = [self.pipeline.tokenizer.encode(s)[-1] for s in stop]
            stopping_criteria = StoppingCriteriaList([StopOnTokens(stop_ids)])
        else:
            stopping_criteria = StoppingCriteriaList()

        # Create TextIteratorStreamer
        streamer = TextIteratorStreamer(self.pipeline.tokenizer, timeout=60.0, skip_prompt=True, skip_special_tokens=True)

        # Prepare input
        inputs = self.pipeline.tokenizer(prompt, return_tensors="pt").to(self.pipeline.device)

        # Set generation parameters
        generation_kwargs = dict(
            **inputs,
            streamer=streamer,
            stopping_criteria=stopping_criteria,
            **pipeline_kwargs,
        )

        # Run generation in background thread
        thread = Thread(target=self.pipeline.model.generate, kwargs=generation_kwargs)
        thread.start()

        # Yield generated text
        for new_text in streamer:
            chunk = GenerationChunk(text=new_text)
            yield chunk
            if run_manager:
                run_manager.on_llm_new_token(new_text, chunk=chunk)