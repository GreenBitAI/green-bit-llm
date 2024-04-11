import argparse
import sys

import torch
import torch.nn as nn

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module='torch.nn.modules.module')

from transformers import PreTrainedTokenizer

from pathlib import Path

# Add the parent directory to sys.path
parent_dir = str(Path(__file__).parent.parent)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from green_bit_llm.model import load, generate

# default value for arguments
DEFAULT_MODEL_PATH = "GreenBitAI/Qwen-1.5-0.5B-layer-mix-bpw-2.2"
DEFAULT_PROMPT = None
DEFAULT_MAX_TOKENS = 100
DEFAULT_TEMP = 0.8
DEFAULT_TOP_P = 0.95
DEFAULT_SEQLEN = 2048
DTYPE = torch.half


def setup_arg_parser():
    """Set up and return the argument parser."""
    parser = argparse.ArgumentParser(description="green-bit-llm inference script")
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL_PATH,
        help="The path to the local model directory or Hugging Face repo.",
    )
    parser.add_argument(
        "--cuda-device-id",
        type=str,
        default="0",
        help="CUDA device IDs",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Enable trusting remote code for tokenizer",
    )
    parser.add_argument(
        "--use-flash-attention-2",
        action="store_true",
        help="Enable using flash attention v2",
    )
    parser.add_argument(
        "--eos-token",
        type=str,
        default=None,
        help="End of sequence token for tokenizer",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=DEFAULT_MAX_TOKENS,
        help="Maximum number of tokens to generate",
    )
    parser.add_argument(
        "--prompt", default=DEFAULT_PROMPT, help="Message to be processed by the model"
    )
    parser.add_argument(
        "--temp", type=float, default=DEFAULT_TEMP, help="Sampling temperature"
    )
    parser.add_argument(
        "--top-p", type=float, default=DEFAULT_TOP_P, help="Sampling top-p"
    )
    parser.add_argument(
        "--seqlen", type=int, default=DEFAULT_SEQLEN, help="Sequence length"
    )
    parser.add_argument(
        "--ignore-chat-template",
        action="store_true",
        help="Use the raw prompt without the tokenizer's chat template.",
    )
    parser.add_argument(
        "--use-default-chat-template",
        action="store_true",
        help="Use the default chat template",
    )
    return parser


def create_device_map(cuda_device_id):
    # TODO: create device map strategy
    #return device_map
    raise NotImplementedError('device map strategy not implemented yet!')


def do_generate(args, model: nn.Module, tokenizer: PreTrainedTokenizer, prompt: str):
    if not args.ignore_chat_template and (
        hasattr(tokenizer, "apply_chat_template")
        and tokenizer.chat_template is not None
    ):
        messages = [{"role": "user", "content": prompt}]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    else:
        prompt = prompt

    generate(
        model,
        tokenizer,
        prompt,
        args.temp,
        args.max_tokens,
        True,
        top_p=args.top_p,
    )


def main(args):

    # Building configs
    tokenizer_config = {"trust_remote_code": True if args.trust_remote_code else None}
    pretrain_model_config = {
        "trust_remote_code": True if args.trust_remote_code else None,
        "attn_implementation": "flash_attention_2" if args.use_flash_attention_2 else None
    }

    if args.eos_token is not None:
        tokenizer_config["eos_token"] = args.eos_token

    model, tokenizer, config = load(
        args.model,
        tokenizer_config=tokenizer_config,
        dtype=DTYPE,
        device_map='auto',
        seqlen=args.seqlen,
        model_config=pretrain_model_config
    )

    if args.use_default_chat_template:
        if tokenizer.chat_template is None:
            tokenizer.chat_template = tokenizer.default_chat_template

    if args.prompt is None:
        while True:
            user_input = input("Input prompt or type 'exit' to quit): ")
            if user_input.lower() in ['exit', 'quit']:
                break
            do_generate(args, model, tokenizer, user_input)
    else:
        do_generate(args, model, tokenizer, args.prompt)


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("Warning: CUDA is needed to run the model.")
        sys.exit(0)

    parser = setup_arg_parser()
    args = parser.parse_args()

    main(args)