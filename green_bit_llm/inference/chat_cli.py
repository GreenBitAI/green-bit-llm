"""
cli chat demo.
Code based on: https://github.com/yanghaojin/FastChat/blob/greenbit/fastchat/serve/cli.py
"""
import argparse
import os
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module='torch.nn.modules.module')

import torch

# Add the parent directory to sys.path
from green_bit_llm.inference.utils import str_to_torch_dtype

try:
    from green_bit_llm.inference.chat_base import chat_loop, SimpleChatIO, RichChatIO
except Exception:
    raise Exception("Error occurred when import chat loop, ChatIO classes.")


def main(args):
    # args setup
    if args.gpus:
        if len(args.gpus.split(",")) < args.num_gpus:
            raise ValueError(
                f"Larger --num-gpus ({args.num_gpus}) than --gpus {args.gpus}!"
            )
        # NOTE that we need to set this before any other cuda operations. Otherwise will not work.
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

    if not torch.cuda.is_available():
        raise Exception("Warning: CUDA is needed to run the model.")

    if args.style == "simple":
        chatio = SimpleChatIO(args.multiline)
    elif args.style == "rich":
        chatio = RichChatIO(args.multiline, args.mouse)
    else:
        raise ValueError(f"Invalid style for console: {args.style}")

    # Building configs
    tokenizer_config = {"trust_remote_code": True if args.trust_remote_code else None}
    pretrain_model_config = {
        "trust_remote_code": True if args.trust_remote_code else None,
        "attn_implementation": "flash_attention_2" if args.use_flash_attention_2 else None
    }

    if args.eos_token is not None:
        tokenizer_config["eos_token"] = args.eos_token

    # chat
    try:
        chat_loop(
            args.model,
            tokenizer_config,
            pretrain_model_config,
            args.seqlen,
            args.device,
            str_to_torch_dtype(args.dtype),
            args.conv_template,
            args.conv_system_msg,
            args.temperature,
            args.repetition_penalty,
            args.max_new_tokens,
            chatio,
            judge_sent_end=args.judge_sent_end,
            debug=args.debug,
            history=not args.no_history,
        )
    except KeyboardInterrupt:
        print("exit...")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="GreenBitAI/Mistral-Instruct-7B-v0.2-layer-mix-bpw-2.2",
        help="The path to the weights. This can be a local folder or a Hugging Face repo ID.",
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["cpu", "cuda"],
        default="cuda",
        help="The device type",
    )
    parser.add_argument(
        "--gpus",
        type=str,
        default='0',
        help="A single GPU like 1 or multiple GPUs like 0,2",
    )
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument(
        "--dtype",
        type=str,
        choices=["float32", "float16"],
        help="Override the default dtype. If not set, it will use float16 on GPU and float32 on CPU.",
        default="float16",
    )
    parser.add_argument(
        "--conv-template", type=str, default=None, help="Conversation prompt template."
    )
    parser.add_argument(
        "--conv-system-msg", type=str, default=None, help="Conversation system message."
    )
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--repetition_penalty", type=float, default=1.0)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--no-history", action="store_true", help="Disables chat history.")
    parser.add_argument(
        "--style",
        type=str,
        default="rich",
        choices=["simple", "rich"],
        help="Display style.",
    )
    parser.add_argument(
        "--multiline",
        action="store_true",
        help="Enable multiline input. Use ESC+Enter for newline.",
    )
    parser.add_argument(
        "--mouse",
        action="store_true",
        help="[Rich Style]: Enable mouse support for cursor positioning.",
    )
    parser.add_argument(
        "--judge-sent-end",
        action="store_true",
        help="Whether enable the correction logic that interrupts the output of sentences due to EOS.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print useful debug information (e.g., prompts)",
    )
    parser.add_argument(
        "--eos-token",
        type=str,
        default=None,
        help="End of sequence token for tokenizer",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Enable trusting remote code",
    )
    parser.add_argument(
        "--seqlen", type=int, default=2048, help="Sequence length"
    )
    parser.add_argument(
        "--use-flash-attention-2",
        action="store_true",
        help="Enable using flash attention v2",
    )
    args = parser.parse_args()
    main(args)