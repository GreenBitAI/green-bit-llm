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
        hidden_states_list = []
        skip_prompt = kwargs.get("skip_prompt", True)
        with_hidden_states = kwargs.get("with_hidden_states", False)

        for i in range(0, len(prompts), self.batch_size):
            batch_prompts = prompts[i: i + self.batch_size]

            # get attention mask of inputs
            inputs = self.pipeline.tokenizer(
                batch_prompts,
                return_tensors="pt",
                padding=True
            ).to(self.pipeline.device)

            # modifies pipeline kwargs for getting hidden states
            if with_hidden_states:
                pipeline_kwargs["output_hidden_states"] = True

            # get llm responses
            outputs = self.pipeline.model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                **pipeline_kwargs,
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
                input_tokens_hs = input_tokens_hs * mask
                # calculates mean
                batch_embeddings = torch.sum(input_tokens_hs, dim=1) / torch.sum(mask, dim=1)
                hidden_states_list.extend([emb.detach() for emb in batch_embeddings])

            generated_texts = self.pipeline.tokenizer.batch_decode(
                outputs.sequences,
                skip_special_tokens=True
            )

            for j, text in enumerate(generated_texts):
                if skip_prompt:
                    text = text[len(batch_prompts[j]):]
                text_generations.append(text)

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