import argparse

DEFAULT_MODEL_PATH = "GreenBitAI/Qwen-1.5-0.5B-layer-mix-bpw-2.2"
DEFAULT_SEQLEN = 2048


def setup_shared_arg_parser(parser_name="Shared argument parser for green-bit-llm scripts"):
    """Set up and return the argument parser with shared arguments."""
    parser = argparse.ArgumentParser(description=parser_name)
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL_PATH,
        help="The path to the local model directory or Hugging Face repo.",
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
        "--seqlen",
        type=int,
        default=DEFAULT_SEQLEN,
        help="Sequence length"
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="output/",
        help="Specify save dir for eval results.",
    )
    parser.add_argument(
        "--save-step",
        type=int,
        default=500,
        help="Specify how many steps to save a checkpoint.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        choices=["float", "half"],
        default="half",
        help="Dtype used in optimizer.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="tatsu-lab/alpaca",
        help="Dataset name for finetuning",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default="DiodeMix",
        help="Optimizer to use: 1. DiodeMix, 2. AdamW8bit"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size"
    )
    parser.add_argument("--weight_decay", type=float, default=0.0)
    return parser
