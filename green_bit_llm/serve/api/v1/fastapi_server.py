import os
import logging
from datetime import datetime
import argparse
import json
import time
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Union, Set

import torch
from fastapi import FastAPI, HTTPException, Depends
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, field_validator, ValidationInfo
from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage, AIMessage
from transformers import AutoTokenizer
import asyncio
from concurrent.futures import ThreadPoolExecutor
# from green_bit_llm.serve.auth import get_api_key_auth, APIKeyAuth
# from starlette.middleware.base import BaseHTTPMiddleware
from dotenv import load_dotenv

from green_bit_llm.langchain import GreenBitPipeline, ChatGreenBit

thread_pool = ThreadPoolExecutor()

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent.parent
LOG_DIR = PROJECT_ROOT / "logs"

# Global configurations
server_config = None
model_provider = None
logger = None

# Global UE confidence scorers
UE_MODELS = {
    "qwen-2.5-7b": "qwen2.5",
    "llama-3-8b": "llama-3"
}
# Global confidence scorers
_confidence_scorers = {}


#====================== Helper classes and methods =====================#

# class RateLimitMiddleware(BaseHTTPMiddleware):
#     """Middleware to handle concurrent request cleanup."""
#
#     def __init__(self, app, auth_handler: APIKeyAuth):
#         super().__init__(app)
#         self.auth_handler = auth_handler
#
#     async def dispatch(self, request, call_next):
#         api_key = request.headers.get("X-Api-Key")
#
#         try:
#             response = await call_next(request)
#             return response
#         finally:
#             if api_key:
#                 await self.auth_handler.rate_limiter.release_concurrent_request(api_key)

def convert_model_name_to_url_path(model_name: str) -> str:
    """
    Convert a model name to a URL-safe path segment.
    Examples:
        "GreenBitAI/Llama-3-8B-instruct-layer-mix-bpw-4.0-mlx" ->
        "GreenBitAI-Llama-3-8B-instruct-layer-mix-bpw-4.0-mlx"
    """
    # Replace forward slashes with dashes
    url_safe_name = model_name.replace("/", "-")

    # Remove any special characters that might cause issues in URLs
    # Keep alphanumeric characters, dashes, and underscores
    url_safe_name = "".join(c for c in url_safe_name
                            if c.isalnum() or c in "-_")

    # Remove any repeated dashes
    while "--" in url_safe_name:
        url_safe_name = url_safe_name.replace("--", "-")

    # Remove leading or trailing dashes
    url_safe_name = url_safe_name.strip("-")

    return url_safe_name


def get_model_endpoint_path(model_name: str, endpoint_type: str) -> str:
    """Generate the full API endpoint path for a given model and endpoint type."""
    safe_name = convert_model_name_to_url_path(model_name)
    return f"/v1/{safe_name}/{endpoint_type}"


def setup_logging():
    """Configure logging for the FastAPI server."""
    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = LOG_DIR / f"server_{timestamp}.log"

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

    logger = logging.getLogger("greenbit_server")
    logger.info(f"Starting GreenBit API server. Log file: {log_file}")
    return logger

def get_model_key(request_model: str) -> str:
    """
    Determine the corresponding model key based on the requested model name

    Args:
        request_model: The model name in the request
    Returns:
        str: The matched model key
    """
    request_model = request_model.lower()

    # First try exact match
    if request_model in UE_MODELS:
        return request_model

    # If no exact match, try standardizing the model name format
    model_families = {
        "qwen-2.5-7b": ["qwen2.5-7b", "qwen-2.5-7b"],
        "llama-3-8b": ["llama3-8b", "llama-3-8b"]
    }

    for standard_name, variants in model_families.items():
        if any(variant in request_model for variant in variants):
            return standard_name

    # If no match is found, raise exception
    raise ValueError(f"Error: Unsupported model: {request_model}")
#================ END of helper classes and methods ======================#

def parse_args():
    parser = argparse.ArgumentParser(
        description="GreenBit FastAPI Server.",
        formatter_class = argparse.RawDescriptionHelpFormatter,
        epilog="""
            Example usage:
               # method 1：read api key from request header
                curl -X POST "http://localhost:8000/v1/model-name/completions" \
                     -H "X-Api-Key: your-api-key" \
                     -H "Content-Type: application/json" \
                     -d '{"prompt": "Hello", "max_tokens": 100}'
                
                # Method 2: Use the API key in the environment variable (no need to specify it in the request header)
                curl -X POST "http://localhost:8000/v1/model-name/completions" \
                     -H "Content-Type: application/json" \
                     -d '{"prompt": "Hello", "max_tokens": 100}'
            """
    )
    # Server configuration
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host address to bind the server")
    parser.add_argument("--port", type=int, default=8000, help="Port number to run the server")

    # Model configuration - support both single and multiple models
    model_group = parser.add_mutually_exclusive_group()
    model_group.add_argument("--model", type=str, help="Path to the model weights, tokenizer, and config")
    model_group.add_argument("--model_list", type=str, nargs="+", help="List of model paths to serve")

    # Additional configurations
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to run the model on")
    parser.add_argument("--max_new_tokens", type=int, default=2048, help="Maximum number of new tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--use_default_chat_template", action="store_true",
                        help="Use the default chat template from the model's tokenizer")
    parser.add_argument("--chat_template", type=str, default="",
                        help="Custom chat template to use instead of default")
    parser.add_argument("--eos_token", type=str, default="<|eot_id|>",
                        help="End of sequence token for tokenizer")
    parser.add_argument("--db_file_path", type=str, default="db/greenbit.db",
                        help="Path to the uncertainty estimation parameters database")
    parser.add_argument("--trust_remote_code", action="store_true",
                        help="Allow loading remote code in tokenizer")
    parser.add_argument("--seqlen", type=int, default=4096, help="Maximum sequence length")
    parser.add_argument("--device_map", type=str, default="auto", help="Device mapping strategy")
    parser.add_argument("--env_file", type=str, default=".env", help="Path to .env file")

    return parser.parse_args()


class ServerConfig:
    def __init__(self, host: str = "127.0.0.1", port: int = 8000, **kwargs):
        self.host = host
        self.port = port
        self.model_config = argparse.Namespace(**kwargs)

        # Store the list of models to serve
        self.models_to_serve: Set[str] = set()
        if kwargs.get("model_list"):
            self.models_to_serve.update(kwargs["model_list"])
        elif kwargs.get("model"):
            self.models_to_serve.add(kwargs["model"])


class ModelProvider:
    def __init__(self, cli_args: argparse.Namespace):
        self.cli_args = cli_args
        self.model_cache = {}  # Cache for loaded models

        try:
            # Load all specified models
            if self.cli_args.model_list:
                for model_path in self.cli_args.model_list:
                    self.load_model(model_path)
            elif self.cli_args.model:
                self.load_model(self.cli_args.model)

        except Exception as e:
            logger.error(f"Failed to initialize ModelProvider: {str(e)}")
            raise

    def load_model(self, model_path: str):
        """Load a model and its tokenizer, with caching."""
        if model_path in self.model_cache:
            return self.model_cache[model_path]

        try:
            # Initialize tokenizer with custom configuration
            tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=True,
                eos_token="<|im_end|>" if model_path.lower().__contains__("qwen") else "<|eot_id|>",
            )

            # Configure chat template
            if self.cli_args.chat_template:
                tokenizer.chat_template = self.cli_args.chat_template
            elif self.cli_args.use_default_chat_template:
                if not tokenizer.chat_template:
                    tokenizer.chat_template = tokenizer.default_chat_template

            model_config = {
                "trust_remote_code": True,
                "attn_implementation": "flash_attention_2"
            }

            tokenizer_config = {"trust_remote_code": True}

            # Initialize pipeline
            pipeline = GreenBitPipeline.from_model_id(
                model_id=model_path,
                model_kwargs={
                    "dtype": torch.half,
                    "device_map": self.cli_args.device_map,
                    "seqlen": self.cli_args.seqlen,
                    "requires_grad": False,
                    "model_config": model_config,
                    "tokenizer_config": tokenizer_config
                },
                pipeline_kwargs={
                    "max_new_tokens": self.cli_args.max_new_tokens,
                    "temperature": self.cli_args.temperature,
                    "eos_token_id": tokenizer.eos_token_id,
                    "do_sample": True
                },
            )

            chat_model = ChatGreenBit(llm=pipeline)

            # Cache the loaded model components
            self.model_cache[model_path] = {
                'chat_model': chat_model,
                'tokenizer': tokenizer,
                'pipeline': pipeline
            }

            logger.info(f"Successfully loaded model {model_path}")
            return self.model_cache[model_path]

        except Exception as e:
            logger.error(f"Failed to load model {model_path}: {str(e)}")
            raise

    def get_model(self, model_path: str):
        """Get a loaded model by its path."""
        if model_path not in self.model_cache:
            return self.load_model(model_path)
        return self.model_cache[model_path]


class CompletionRequest(BaseModel):
    """
    Example requests:
    # Single completion
    request = CompletionRequest(
        model="model_name",
        prompt="Hello world"
    )
    # Batch completion
    request = CompletionRequest(
        model="model_name",
        prompt=["Hello world", "How are you"]
    )
    """
    model: str
    prompt: Union[str, List[str]]
    max_tokens: int = 100
    temperature: float = 1.0
    top_p: float = 1.0
    stream: bool = False
    stop: Optional[Union[str, List[str]]] = None
    logit_bias: Optional[Dict[str, float]] = None
    repetition_penalty: float = 1.0
    repetition_context_size: int = 20
    with_hidden_states: bool = False
    remote_score: bool = True
    api_key: str = ""

    @field_validator('prompt')
    @classmethod
    def validate_prompt(cls, v):
        if isinstance(v, list):
            if not v:
                raise ValueError("Prompt list cannot be empty")
            if not all(isinstance(p, str) for p in v):
                raise ValueError("All prompts must be strings")
            if any(not p.strip() for p in v):
                raise ValueError("Prompts cannot be empty strings")
        elif not v.strip():
            raise ValueError("Prompt cannot be empty")
        return v

    @field_validator('stream', 'with_hidden_states')
    @classmethod
    def validate_stream_with_batch(cls, v, info: ValidationInfo):
        prompt = info.data.get('prompt')
        if v and isinstance(prompt, list) and len(prompt) > 1:
            raise ValueError("Streaming is not supported with batch requests")
        return v


class ChatCompletionRequest(BaseModel):
    """
    Example requests:
    # Single chat
    request = ChatCompletionRequest(
        model="model_name",
        messages=[
            {"role": "user", "content": "Hello"}
        ]
    )
    # Batch chat
    request = ChatCompletionRequest(
        model="model_name",
        messages=[
            [{"role": "user", "content": "Hello"}],
            [{"role": "user", "content": "Hi"}]
        ]
    )
    """
    model: str
    messages: Union[List[Dict[str, str]], List[List[Dict[str, str]]]]
    max_tokens: int = 100
    temperature: float = 1.0
    top_p: float = 1.0
    stream: bool = False
    stop: Optional[Union[str, List[str]]] = None
    logit_bias: Optional[Dict[str, float]] = None
    repetition_penalty: float = 1.0
    repetition_context_size: int = 20
    with_hidden_states: bool = False
    remote_score: bool = True
    api_key: str = ""
    enable_thinking: Optional[bool] = None

    @field_validator('messages')
    @classmethod
    def validate_messages(cls, v):
        if not v:
            raise ValueError("Messages cannot be empty")

        if isinstance(v[0], dict):
            cls._validate_single_conversation(v)
        elif isinstance(v[0], list):
            if not all(isinstance(conv, list) for conv in v):
                raise ValueError("All items in messages must be lists of messages")
            for conv in v:
                cls._validate_single_conversation(conv)
        else:
            raise ValueError("Invalid messages format")
        return v

    @staticmethod
    def _validate_single_conversation(messages):
        if not messages:
            raise ValueError("Conversation cannot be empty")

        for msg in messages:
            if not isinstance(msg, dict):
                raise ValueError("Each message must be a dictionary")

            if 'role' not in msg or 'content' not in msg:
                raise ValueError("Each message must have 'role' and 'content'")

            if msg['role'] not in ['system', 'user', 'assistant']:
                raise ValueError("Message role must be 'system', 'user', or 'assistant'")

            if not isinstance(msg['content'], str) or not msg['content'].strip():
                raise ValueError("Message content must be a non-empty string")

    @field_validator('stream', 'with_hidden_states')
    @classmethod
    def validate_stream_with_batch(cls, v, info: ValidationInfo):
        messages = info.data.get('messages', [])
        is_batch = isinstance(messages, list) and messages and isinstance(messages[0], list)
        if v and is_batch and len(messages) > 1:
            raise ValueError("Streaming is not supported with batch requests")
        return v


async def stream_completion(request: CompletionRequest, chat_model: ChatGreenBit):
    created = int(time.time())
    request_id = f"cmpl-{uuid.uuid4()}"

    try:
        # Prepare and wrap generation parameters
        generation_kwargs = {
            "temperature": request.temperature,
            "max_new_tokens": request.max_tokens
        }
        wrapped_kwargs = {
            "pipeline_kwargs": generation_kwargs
        }

        # Get tokenizer for counting tokens
        tokenizer = chat_model.llm.pipeline.tokenizer

        # Count input tokens
        input_tokens = len(tokenizer.encode(request.prompt))
        output_tokens = 0
        is_first_chunk = True

        for chunk in chat_model.llm.stream(request.prompt, **wrapped_kwargs):
            # Count tokens in this chunk
            chunk_tokens = len(tokenizer.encode(chunk.text))
            output_tokens += chunk_tokens

            response = {
                "id": request_id,
                "object": "text_completion",
                "created": created,
                "model": request.model,
                "choices": [
                    {
                        "text": chunk.text,
                        "index": 0,
                        "logprobs": None,
                        "finish_reason": None,
                    }
                ]
            }

            # Add usage info in first chunk
            if is_first_chunk:
                response["usage"] = {
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "total_tokens": input_tokens + output_tokens
                }
                is_first_chunk = False

            yield f"data: {json.dumps(response)}\n\n"

        # Final chunk with complete usage stats
        response = {
            "id": request_id,
            "object": "text_completion",
            "created": created,
            "model": request.model,
            "choices": [
                {
                    "text": "",
                    "index": 0,
                    "logprobs": None,
                    "finish_reason": "max_tokens" if output_tokens == request.max_tokens else "stop",
                }
            ],
            "usage": {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens
            }
        }
        yield f"data: {json.dumps(response)}\n\n"
        yield "data: [DONE]\n\n"
    except Exception as e:
        logger.error(f"Streaming completion failed: {str(e)}")
        raise


async def stream_chat_completion(request: ChatCompletionRequest, chat_model: ChatGreenBit,
                               messages_list: List[List[BaseMessage]]):
    created = int(time.time())
    request_id = f"chatcmpl-{uuid.uuid4()}"

    try:
        # Prepare and wrap generation parameters
        generation_kwargs = {
            "temperature": request.temperature,
            "max_new_tokens": request.max_tokens
        }

        if request.enable_thinking is not None:
            generation_kwargs["enable_thinking"] = request.enable_thinking

        wrapped_kwargs = {
            "pipeline_kwargs": generation_kwargs
        }

        # Get tokenizer for counting tokens
        tokenizer = chat_model.llm.pipeline.tokenizer

        output_tokens = 0
        is_first_chunk = True

        for chunk in chat_model.stream(messages_list[0], **wrapped_kwargs):
            # Count tokens in this chunk
            chunk_tokens = len(tokenizer.encode(chunk.message.content))
            output_tokens += chunk_tokens

            response = {
                "id": request_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": request.model,
                "choices": [
                    {
                        "delta": {
                            "role": "assistant",
                            "content": chunk.message.content
                        },
                        "index": 0,
                        "finish_reason": None,
                    }
                ]
            }
            # Add usage info in first chunk
            if is_first_chunk:
                input_tokens = 0
                for message in messages_list[0]:
                    input_tokens += len(tokenizer.encode(message.content))

                response["usage"] = {
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "total_tokens": input_tokens + output_tokens
                }
                is_first_chunk = False

            yield f"data: {json.dumps(response)}\n\n"

        # Final chunk with complete usage stats
        response = {
            "id": request_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": request.model,
            "choices": [
                {
                    "delta": {
                        "role": "assistant",
                        "content": ""
                    },
                    "index": 0,
                    "finish_reason": "max_tokens" if output_tokens == request.max_tokens else "stop",
                }
            ],
            "usage": {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens
            }
        }
        yield f"data: {json.dumps(response)}\n\n"
        yield "data: [DONE]\n\n"
    except Exception as e:
        logger.error(f"Streaming chat completion failed: {str(e)}")
        raise


async def generate_completion(request: CompletionRequest, chat_model: ChatGreenBit):
    created = int(time.time())
    request_id = f"cmpl-{uuid.uuid4()}"

    try:
        generation_kwargs = {
            "temperature": request.temperature,
            "max_new_tokens": request.max_tokens
        }

        # Wrap generation parameters
        wrapped_kwargs = {
            "pipeline_kwargs": generation_kwargs,
            "with_hidden_states": request.with_hidden_states
        }

        prompts = request.prompt if isinstance(request.prompt, list) else [request.prompt]

        # thread pool
        llm_result = await asyncio.get_event_loop().run_in_executor(
            thread_pool,
            lambda: chat_model.llm.generate(
                prompts=prompts,
                **wrapped_kwargs
            )
        )

        # handle results
        choices = []
        for idx, generations in enumerate(llm_result.generations):
            gen_info = generations[0].generation_info or {}
            hidden_states = gen_info.get("hidden_states")
            score = None

            if request.with_hidden_states and hidden_states is not None:
                if request.remote_score:
                    model_key = get_model_key(request.model)
                    scorer = _confidence_scorers.get(model_key)
                    if scorer:
                        # thread pool for scoring
                        scores = await asyncio.get_event_loop().run_in_executor(
                            thread_pool,
                            scorer.calculate_confidence,
                            hidden_states
                        )
                        score = scores if isinstance(scores, list) else [scores]
                    hidden_states=None
                else:
                    hidden_states = await asyncio.get_event_loop().run_in_executor(
                        thread_pool,
                        lambda: hidden_states.cpu().tolist()
                    )

            # Let tokenizer determine if max_tokens was reached
            tokenizer = chat_model.llm.pipeline.tokenizer
            generated_text = generations[0].text
            finish_reason = "length" if len(tokenizer.encode(generated_text)) >= request.max_tokens else "stop"

            choices.append({
                "text": generations[0].text,
                "index": idx,
                "logprobs": None,
                "finish_reason": finish_reason,
                "hidden_states": hidden_states,
                "confidence_score": score[idx] if score else None
            })

        async def calculate_tokens():
            try:
                tokenizer = chat_model.llm.pipeline.tokenizer
                # thread pool
                prompt_tokens = await asyncio.gather(*[
                    asyncio.get_event_loop().run_in_executor(
                        thread_pool,
                        lambda p=p: len(tokenizer.encode(p))
                    )
                    for p in prompts
                ])
                completion_tokens = await asyncio.gather(*[
                    asyncio.get_event_loop().run_in_executor(
                        thread_pool,
                        lambda c=c: len(tokenizer.encode(c["text"]))
                    )
                    for c in choices
                ])
                return {
                    "input_tokens": sum(prompt_tokens),
                    "output_tokens": sum(completion_tokens),
                    "total_tokens": sum(prompt_tokens) + sum(completion_tokens)
                }
            except Exception as e:
                logger.warning(f"Error calculating exact token counts: {str(e)}. Using approximate count.")
                prompt_tokens = sum(len(p.split()) for p in prompts)
                completion_tokens = sum(len(c["text"].split()) for c in choices)
                return {
                    "input_tokens": prompt_tokens,
                    "output_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens
                }

        usage = await calculate_tokens()

        return {
            "id": request_id,
            "object": "text_completion",
            "created": created,
            "model": request.model,
            "choices": choices,
            "usage": usage
        }
    except Exception as e:
        logger.error(f"Completion generation failed: {str(e)}")
        if isinstance(e, torch.cuda.OutOfMemoryError):
            raise HTTPException(status_code=503,
                              detail="GPU memory exhausted, please try with shorter input or smaller batch")
        elif isinstance(e, ValueError):
            raise HTTPException(status_code=400, detail=str(e))
        raise HTTPException(status_code=500, detail=str(e))


async def generate_chat_completion(request: ChatCompletionRequest, chat_model: ChatGreenBit,
                                 messages_list: List[List[BaseMessage]]):
    created = int(time.time())
    request_id = f"chatcmpl-{uuid.uuid4()}"

    try:
        generation_kwargs = {
            "temperature": request.temperature,
            "top_p": request.top_p,
            "max_new_tokens": request.max_tokens
        }

        if request.enable_thinking is not None:
            generation_kwargs["enable_thinking"] = request.enable_thinking

        # Wrap generation parameters
        wrapped_kwargs = {
            "pipeline_kwargs": generation_kwargs,
            "with_hidden_states": request.with_hidden_states
        }

        prompt = chat_model._prepare_prompt(messages_list[0], **generation_kwargs)

        # Generate using thread pool
        llm_result = await asyncio.get_event_loop().run_in_executor(
            thread_pool,
            lambda: chat_model.llm.generate(
                prompts=[prompt],
                **wrapped_kwargs
            )
        )

        # handle results
        choices = []
        for idx, generations in enumerate(llm_result.generations):
            generation = generations[0]
            gen_info = generation.generation_info or {}
            hidden_states = gen_info.get("hidden_states")
            score = None

            if request.with_hidden_states and hidden_states is not None:
                if request.remote_score:
                    model_key = get_model_key(request.model)
                    scorer = _confidence_scorers.get(model_key)
                    if scorer:
                        # thread pool
                        scores = await asyncio.get_event_loop().run_in_executor(
                            thread_pool,
                            scorer.calculate_confidence,
                            hidden_states
                        )
                        score = scores if isinstance(scores, list) else [scores]

                    hidden_states = None
                else:
                    # CPU ops in thread pool
                    hidden_states = await asyncio.get_event_loop().run_in_executor(
                        thread_pool,
                        lambda: hidden_states.cpu().tolist()
                    )

            # Let tokenizer determine if max_tokens was reached
            tokenizer = chat_model.llm.pipeline.tokenizer
            generated_text = generation.text
            finish_reason = "length" if len(tokenizer.encode(generated_text)) >= request.max_tokens else "stop"

            choices.append({
                "message": {
                    "role": "assistant",
                    "content": generated_text,
                    "hidden_states": hidden_states,
                    "confidence_score": score[idx] if score else None
                },
                "index": idx,
                "finish_reason": finish_reason
            })

        async def calculate_tokens():
            try:
                tokenizer = chat_model.llm.pipeline.tokenizer

                # input tokens
                input_tokens = 0
                for message in messages_list[0]:
                    input_tokens += len(tokenizer.encode(message.content))

                completion_tokens = await asyncio.gather(*[
                    asyncio.get_event_loop().run_in_executor(
                        thread_pool,
                        lambda c=c: len(tokenizer.encode(c["message"]["content"]))
                    )
                    for c in choices
                ])
                return {
                    "input_tokens": input_tokens,
                    "output_tokens": sum(completion_tokens),
                    "total_tokens": input_tokens + sum(completion_tokens)
                }
            except Exception as e:
                logger.warning(f"Error calculating exact token counts: {str(e)}. Using approximate count.")
                # fall back
                input_tokens = sum(len(msg.content.split()) for msg in messages_list[0])
                completion_tokens = sum(len(c["message"]["content"].split()) for c in choices)
                return {
                    "input_tokens": input_tokens,
                    "output_tokens": completion_tokens,
                    "total_tokens": input_tokens + completion_tokens
                }

        usage = await calculate_tokens()

        return {
            "id": request_id,
            "object": "chat.completion",
            "created": created,
            "model": request.model,
            "choices": choices,
            "usage": usage
        }
    except Exception as e:
        logger.error(f"Chat completion generation failed: {str(e)}")
        if isinstance(e, torch.cuda.OutOfMemoryError):
            raise HTTPException(status_code=503,
                              detail="GPU memory exhausted, please try with shorter input or smaller batch")
        elif isinstance(e, ValueError):
            raise HTTPException(status_code=400, detail=str(e))
        raise HTTPException(status_code=500, detail=str(e))

def create_app(args):
    """Create and configure the FastAPI application with routes."""

    # Initialize logging
    global logger
    logger = setup_logging()

    # CUDA specific optimization
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    # Initialize confidence scorers
    try:
        from green_bit_llm.routing import ConfidenceScorer
        for model_family, model_id in UE_MODELS.items():
            scorer = ConfidenceScorer(
                parameters_path=args.db_file_path,
                model_id=model_id
            )
            _confidence_scorers[model_family] = scorer
    except Exception as e:
        logger.error(f"Error loading confidence scorers: {str(e)}")

    app = FastAPI(
        title="GreenBit API",
        description="API using GreenBit models",
    )
    # Load environment variables
    if args.env_file and Path(args.env_file).exists():
        load_dotenv(args.env_file)

    # Update DB path from environment if provided
    db_path = os.getenv("LIBRA_DB_PATH", args.db_file_path)

    # We disable user auth for now
    # # Initialize auth handler
    # auth_handler = APIKeyAuth(db_path)
    # app.add_middleware(RateLimitMiddleware, auth_handler=auth_handler)

    # Add logging middleware
    @app.middleware("http")
    async def log_requests(request, call_next):
        start_time = time.time()
        try:
            response = await call_next(request)
            duration = time.time() - start_time
            logger.info(
                f"Request: {request.method} {request.url.path} "
                f"Status: {response.status_code} "
                f"Duration: {duration:.2f}s"
            )
            return response
        except Exception as e:
            logger.error(f"Request failed: {str(e)}")
            raise

    # Create server config with all arguments
    server_config = ServerConfig(
        host=args.host,
        port=args.port,
        model=args.model,
        model_list=args.model_list,
        device=args.device,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        use_default_chat_template=args.use_default_chat_template,
        chat_template=args.chat_template,
        eos_token=args.eos_token,
        trust_remote_code=args.trust_remote_code,
        env_file=args.env_file,
        db_file_path=args.db_file_path,
        seqlen=args.seqlen,
        device_map=args.device_map
    )

    # Initialize model provider
    model_provider = ModelProvider(server_config.model_config)

    # Helper function to create endpoints for a specific model
    def create_model_endpoints(model_path: str):
        completion_path = get_model_endpoint_path(model_path, "completions")
        chat_completion_path = get_model_endpoint_path(model_path, "chat/completions")

        logger.info(f"Creating endpoints for model {model_path}:")
        logger.info(f"  - Completion endpoint: {completion_path}")
        logger.info(f"  - Chat completion endpoint: {chat_completion_path}")

        @app.post(completion_path)
        async def create_completion(
            request: CompletionRequest,
            # We disable user auth for now
            # user_info: dict = Depends(get_api_key_auth)
        ):
            try:
                # We disable user auth for now
                # # Estimate total tokens
                # if isinstance(request.prompt, list):
                #     estimated_tokens = sum(len(p.split()) for p in request.prompt) + request.max_tokens * len(
                #         request.prompt)
                # else:
                #     estimated_tokens = len(request.prompt.split()) + request.max_tokens
                #
                # # Check permissions
                # auth_handler.check_permissions(user_info, "completion")
                #
                # # Check token limit
                # auth_handler.check_token_limit(user_info, request.max_tokens)
                #
                # # Check rate limits with token estimate
                # await auth_handler.check_rate_limits(
                #     request.api_key,
                #     user_info,
                #     estimated_tokens
                # )

                model_components = model_provider.get_model(model_path)
                chat_model = model_components['chat_model']

                if request.stream:
                    return StreamingResponse(
                        stream_completion(request, chat_model),
                        media_type="text/event-stream"
                    )
                else:
                    result = await generate_completion(request, chat_model)
                    return JSONResponse(result)
            except Exception as e:
                logger.error(f"Completion request failed for {model_path}: {str(e)}")
                raise HTTPException(status_code=500, detail=str(e))

        @app.post(chat_completion_path)
        async def create_chat_completion(
            request: ChatCompletionRequest,
            # We disable user auth for now
            # user_info: dict = Depends(get_api_key_auth)
        ):
            try:
                # We disable user auth for now
                # # Check permissions
                # auth_handler.check_permissions(user_info, "chat")
                #
                # # Check token limit
                # auth_handler.check_token_limit(user_info, request.max_tokens)
                #
                # Rough token estimation for chat
                # estimated_tokens = sum(
                #     len(msg["content"].split())
                #     for msg in (request.messages if isinstance(request.messages[0], dict)
                #                 else [item for sublist in request.messages for item in sublist])
                # ) + request.max_tokens
                #
                # # Check rate limits with token estimate
                # await auth_handler.check_rate_limits(
                #     request.api_key,
                #     user_info,
                #     estimated_tokens
                # )

                model_components = model_provider.get_model(model_path)
                chat_model = model_components['chat_model']

                if isinstance(request.messages[0], dict):  # single chat
                    messages_list = []
                    langchain_messages = []
                    for msg in request.messages:
                        if msg["role"] == "system":
                            langchain_messages.append(SystemMessage(content=msg["content"]))
                        elif msg["role"] == "user":
                            langchain_messages.append(HumanMessage(content=msg["content"]))
                        elif msg["role"] == "assistant":
                            langchain_messages.append(AIMessage(content=msg["content"]))
                    messages_list = [langchain_messages]
                else:  # batch
                    messages_list = []
                    for conversation in request.messages:
                        langchain_messages = []
                        for msg in conversation:
                            if msg["role"] == "system":
                                langchain_messages.append(SystemMessage(content=msg["content"]))
                            elif msg["role"] == "user":
                                langchain_messages.append(HumanMessage(content=msg["content"]))
                            elif msg["role"] == "assistant":
                                langchain_messages.append(AIMessage(content=msg["content"]))
                        messages_list.append(langchain_messages)

                if request.stream:
                    return StreamingResponse(
                        stream_chat_completion(request, chat_model, messages_list),
                        media_type="text/event-stream"
                    )
                else:
                    result = await generate_chat_completion(request, chat_model, messages_list)
                    return JSONResponse(result)
            except Exception as e:
                logger.error(f"Chat completion request failed for {model_path}: {str(e)}")
                raise HTTPException(status_code=500, detail=str(e))

    # Create endpoints for each model
    for model_path in server_config.models_to_serve:
        create_model_endpoints(model_path)

    # Add root endpoint for API information
    @app.get("/")
    async def root():
        try:
            models = list(server_config.models_to_serve)
            endpoints = []

            for model in models:
                endpoints.extend([
                    get_model_endpoint_path(model, "completions"),
                    get_model_endpoint_path(model, "chat/completions")
                ])

            return {
                "api": "GreenBit API",
                "version": "1.0",
                "models": models,
                "endpoints": endpoints
            }
        except Exception as e:
            logger.error(f"Root endpoint request failed: {str(e)}")
            raise HTTPException(status_code=500, detail="Internal server error")

    @app.get("/health")
    async def health_check():
        try:
            # check CUDA available
            cuda_available = torch.cuda.is_available()
            if not cuda_available:
                raise HTTPException(status_code=503, detail="CUDA not available")
            return {
                "status": "healthy",
                "cuda_available": True
            }
        except Exception as e:
            raise HTTPException(status_code=503, detail=str(e))

    @app.exception_handler(Exception)
    async def global_exception_handler(request, exc):
        logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"message": "Internal server error"}
        )

    return app, server_config, logger

def main():
    logger = None
    try:
        import uvicorn
        args = parse_args()
        app, server_config, logger = create_app(args)
        logger.info(f"Starting server on {server_config.host}:{server_config.port}")
        uvicorn.run(app, host=server_config.host, port=server_config.port)
    except Exception as e:
        if logger:
            logger.error(f"Server startup failed: {str(e)}")
        else:
            print(f"Server startup failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()
