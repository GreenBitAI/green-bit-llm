from green_bit_llm.inference.sim_gen import DTYPE
from .base import BaseInferenceBackend
import os

import torch
import torch.nn as nn

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module='torch.nn.modules.module')

from transformers import PreTrainedTokenizer

from green_bit_llm.common import generate, load
from green_bit_llm.args_parser import setup_shared_arg_parser

# default value for arguments
DEFAULT_PROMPT = None
DEFAULT_MAX_TOKENS = 100
DEFAULT_TEMP = 0.8
DEFAULT_TOP_P = 0.95
DTYPE = torch.half

class GBLLMInferenceBackend(BaseInferenceBackend):
    def __init__(self, model_path, **kwargs):
        # Building configs
        tokenizer_config = {"trust_remote_code": True if kwargs.get("trust_remote_code") else None}
        pretrain_model_config = {
            "trust_remote_code": True if kwargs.get("trust_remote_code") else None,
            "attn_implementation": "flash_attention_2" if kwargs.get("use_flash_attention_2") else None
        }
        if kwargs.get("eos_token") is not None:
            tokenizer_config["eos_token"] = kwargs.get("eos_token")
        
        self.model, self.tokenizer, config = load(
            model_path,
            tokenizer_config=tokenizer_config,
            dtype=kwargs.get("dtype", DTYPE),
            device_map=kwargs.get("auto", "auto"),
            seqlen=kwargs.get("seqlen", 2048),
            model_config=pretrain_model_config,
            requires_grad=False
        )
        
    def generate(self, prompt, params=None):
        if params == None:
            params = {}
        if isinstance(prompt, str):
            prompt = [prompt]
        for prom in prompt:
            generate(
                self.model,
                self.tokenizer,
                prom,
                params.get("temperature", DEFAULT_TEMP),
                params.get("max_tokens", DEFAULT_MAX_TOKENS),
                True,
                params.get("top_p", DEFAULT_TOP_P),
            )