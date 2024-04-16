import argparse
import sys
import json

import torch
import torch.nn as nn

import transformers
from transformers import PreTrainedTokenizer, TrainingArguments
from datasets import load_dataset
from trl import SFTTrainer

from green_bit_llm.model import load

import warnings

warnings.filterwarnings('ignore')

ENGINE_AVAILABLE = True
try:
    from bitorch_engine.optim import DiodeMix
except ModuleNotFoundError as e:
    ENGINE_AVAILABLE = False
    raise Exception("Error: Bitorch Engine optimizer (DiodeMix) are not available.")

# default value for arguments
DEFAULT_MODEL_PATH = "GreenBitAI/Qwen-1.5-4B-layer-mix-bpw-2.2"
DEFAULT_SEQLEN = 2048
DEFAULT_RANDOM_SEED = 0
DTYPE = torch.half

def setup_arg_parser():
    """Set up and return the argument parser."""
    parser = argparse.ArgumentParser(description="green-bit-llm finetune script")
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_RANDOM_SEED,
        help="The random seed for data loader.",
    )
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
        default="<|im_end|>",
        help="End of sequence token for tokenizer",
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
        default="../log/",
        help="Specify save dir for eval results.",
    )
    return parser


def create_device_map(cuda_device_id):
    ids = cuda_device_id.split(',')
    # Create strings in the format "cuda:x" for each ID and put them into the collection
    device_map = {f"cuda:{id}" for id in ids}
    return device_map

def main(args):

    # Building configs
    tokenizer_config = {"trust_remote_code": True if args.trust_remote_code else None}
    pretrain_model_config = {
        "trust_remote_code": True if args.trust_remote_code else None,
        "use_flash_attention_2": True if args.use_flash_attention_2 else None
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
        requires_grad=True,
    )
    
    model.train()

    dataset = load_dataset("imdb", split="train")

    args = TrainingArguments(output_dir="./output",
                            gradient_checkpointing=True,
                            auto_find_batch_size=True,
                            )

    optimizer = DiodeMix(model.parameters(), lr=5e-6)

    optimizers = (optimizer, None)

    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=dataset,
        dataset_text_field="text",
        optimizers=optimizers,
        #max_seq_length=args.seqlen,
    )

    trainer.train()

if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("Warning: CUDA is needed to run the model.")
        sys.exit(0)

    parser = setup_arg_parser()
    args = parser.parse_args()

    main(args)
