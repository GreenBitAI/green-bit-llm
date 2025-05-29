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


def setup_arg_parser():
    """Set up and return the argument parser."""
    parser = setup_shared_arg_parser("green-bit-llm inference script")

    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument(
        "--gpus",
        type=str,
        default='0',
        help="A single GPU like 1 or multiple GPUs like 0,2",
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
        "--ignore-chat-template",
        action="store_true",
        help="Use the raw prompt without the tokenizer's chat template.",
    )
    parser.add_argument(
        "--use-default-chat-template",
        action="store_true",
        help="Use the default chat template",
    )
    parser.add_argument(
        "--enable-thinking",
        action="store_true",
        help="Enable thinking mode for Qwen-3 models.",
    )
    return parser


def do_generate(args, model: nn.Module, tokenizer: PreTrainedTokenizer, prompt: str, enable_thinking: bool):
    """
    This function generates text based on a given prompt using a model and tokenizer.
    It handles optional pre-processing with chat templates if specified in the arguments.
    """
    if not args.ignore_chat_template and (
        hasattr(tokenizer, "apply_chat_template")
        and tokenizer.chat_template is not None
    ):
        messages = [{"role": "user", "content": prompt}]
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking
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
        top_p=args.top_p
    )


def main(args):

    if args.gpus:
        if len(args.gpus.split(",")) < args.num_gpus:
            raise ValueError(
                f"Larger --num-gpus ({args.num_gpus}) than --gpus {args.gpus}!"
            )
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

    if not torch.cuda.is_available():
        raise Exception("Warning: CUDA is needed to run the model.")

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
        model_config=pretrain_model_config,
        requires_grad=False
    )

    if args.use_default_chat_template:
        if tokenizer.chat_template is None:
            tokenizer.chat_template = tokenizer.default_chat_template

    if args.prompt is None:
        while True:
            user_input = input("Input prompt or type 'exit' to quit): ")
            if user_input.lower() in ['exit', 'quit']:
                break
            do_generate(args, model, tokenizer, user_input, args.enable_thinking)
    else:
        do_generate(args, model, tokenizer, args.prompt, args.enable_thinking)


if __name__ == "__main__":
    parser = setup_arg_parser()
    args = parser.parse_args()

    main(args)