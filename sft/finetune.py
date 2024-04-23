import argparse
import sys
import os

import torch

from transformers import PreTrainedTokenizer, TrainingArguments
from datasets import load_dataset
from trl import SFTTrainer

from green_bit_llm.model import load

import warnings

warnings.filterwarnings('ignore')

try:
    from bitorch_engine.optim import DiodeMix
except ModuleNotFoundError as e:
    raise Exception("Error: Bitorch Engine optimizer (DiodeMix) are not available.")

from .optim import AdamW8bit

# default value for arguments
DEFAULT_MODEL_PATH = "GreenBitAI/Qwen-1.5-1.8B-layer-mix-bpw-3.0"
DEFAULT_SEQLEN = 512
DEFAULT_RANDOM_SEED = 0
DEFAULT_LR = 1e-5
DEFAULT_LR_GALORE = 1e-4
DEFAULT_LR_ADAMW8BIT = 5e-3
DEFAULT_BETAS = (0.9, 0.999)

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
        "--dtype",
        type=str,
        choices=["float", "half"],
        default="half",
        help="Dtype used in optimizer.",
    )
    # GaLore parameters
    parser.add_argument(
        "--galore",
        action="store_true",
        help="Enable using galore",
    )
    parser.add_argument("--rank", type=int, default=128)
    parser.add_argument("--update_proj_gap", type=int, default=200)
    parser.add_argument("--galore_scale", type=float, default=0.25)
    parser.add_argument("--proj_type", type=str, default="std")

    parser.add_argument(
        "--dataset",
        type=str,
        default="tatsu-lab/alpaca",
        help="Dataset name for finetuning",
    )
    # qweight related
    parser.add_argument(
        "--tune-qweight-only",
        action="store_true",
        help="Set whether to adjust only the low-bit qweight and keep the regular parameters unchanged during the training process.",
    )
    parser.add_argument(
        "--lr-2bit",
        type=float,
        default=-1.0,
        help="Learning rate for 2-bit qweight."
    )
    parser.add_argument(
        "--lr-4bit",
        type=float,
        default=-1.0,
        help="Learning rate for 4-bit qweight."
    )
    parser.add_argument(
        "--lr-fp",
        type=float,
        default=DEFAULT_LR,
        help="Learning rate for full-precision weight."
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default="DiodeMix",
        help="Optimizer to use: 1. DiodeMix, 2. AdamW8bit"
    )
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size"
    )
    return parser


def str_to_torch_dtype(dtype: str):
    if dtype is None:
        return None
    elif dtype == "float":
        return torch.float
    elif dtype == "half":
        return torch.float16
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")


def get_learning_rate(lr_bit, galore, default_lr_galore, default_lr):
    if lr_bit > 0:
        return lr_bit
    return default_lr_galore if galore else default_lr


def create_param_groups(model, args: argparse.ArgumentParser):
    """
    Create parameter groups based on the bit-width of quantized weights in the model.
    This function categorizes parameters into groups with different learning rates and beta values
    for optimizers.

    Args:
        model (nn.Module): The neural network model.
        args (argparse.ArgumentParser): Command line arguments for additional parameters.

    Returns:
        List[dict]: A list of dictionaries where each dictionary contains a parameter group.
    """
    params_2_bit = []
    params_4_bit = []

    from bitorch_engine.layers.qlinear.nbit import MPQLinearBase

    for module_name, module in model.named_modules():
        if issubclass(type(module), MPQLinearBase):
            if module.w_bit == 2:
                params_2_bit.append(module.qweight)
            elif module.w_bit == 4:
                params_4_bit.append(module.qweight)
            else:
                raise Exception(f"Error: Invalid qweight bit width: '{module.w_bit}'.")

    id_2bit_params = [id(p) for p in params_2_bit]
    id_4bit_params = [id(p) for p in params_4_bit]
    # Concatenate IDs to form a single list
    excluded_ids = id_2bit_params + id_4bit_params

    # Create list of regular parameters excluding 2-bit and 4-bit params
    params_regular = [p for p in model.parameters() if id(p) not in excluded_ids]

    lr_2 = get_learning_rate(
        args.lr_2bit,
        args.galore,
        DEFAULT_LR_ADAMW8BIT if 'adamw8bit' in args.optimizer else DEFAULT_LR_GALORE,
        DEFAULT_LR)
    lr_4 = get_learning_rate(
        args.lr_4bit,
        args.galore,
        DEFAULT_LR_ADAMW8BIT if 'adamw8bit' in args.optimizer else DEFAULT_LR_GALORE,
        DEFAULT_LR)

    params_group_2bit = {'params': params_2_bit, 'lr': lr_2, 'betas': DEFAULT_BETAS}
    params_group_4bit = {'params': params_4_bit, 'lr': lr_4, 'betas': DEFAULT_BETAS}
    params_group_regular = {'params': params_regular, 'lr': args.lr_fp, 'betas': DEFAULT_BETAS}

    # Optionally add extra settings from command line arguments
    if args.galore:
        galore_settings = {
            'rank': args.rank,
            'update_proj_gap': args.update_proj_gap,
            'scale': args.galore_scale,
            'proj_type': args.proj_type
        }
        params_group_2bit.update(galore_settings)
        params_group_4bit.update(galore_settings)

    param_groups = [
        params_group_2bit,
        params_group_4bit
    ]
    if not args.tune_qweight_only:
        param_groups.append(params_group_regular)

    return param_groups


def main(args):

    # Building configs
    tokenizer_config = {"trust_remote_code": True if args.trust_remote_code else None}
    pretrain_model_config = {
        "trust_remote_code": True if args.trust_remote_code else None,
        "use_flash_attention_2": True if args.use_flash_attention_2 else None
    }

    model, tokenizer, config = load(
        args.model,
        tokenizer_config=tokenizer_config,
        device_map='auto',
        seqlen=args.seqlen,
        model_config=pretrain_model_config,
        requires_grad=True,
    )

    param_groups = create_param_groups(model, args)

    model.train()

    dataset = load_dataset(args.dataset, split="train")
    
    args.save_dir = args.save_dir + args.model

    train_args = TrainingArguments(
                    output_dir=args.save_dir,
                    gradient_checkpointing=True,
                    # auto_find_batch_size=True,
                    per_device_train_batch_size=args.batch_size,
                    logging_steps=1,
                    save_steps=50,
                    max_grad_norm=0, # NOTE: max_grad_norm MUST be <= 0 or None, otherwise raise dtype error due to the Int dtype of qweight.
                )

    # Optimizer
    if 'adamw8bit' in args.optimizer.lower():
        optimizer = AdamW8bit(param_groups, weight_decay=args.weight_decay, dtype=str_to_torch_dtype(args.dtype))
    elif 'diodemix' in args.optimizer.lower():
        optimizer = DiodeMix(param_groups, dtype=str_to_torch_dtype(args.dtype))

    optimizers = (optimizer, None)

    # Trainer
    trainer = SFTTrainer(
        model=model,
        args=train_args,
        train_dataset=dataset,
        dataset_text_field="text",
        optimizers=optimizers,
        max_seq_length=args.seqlen,
    )

    trainer.train()

if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("Warning: CUDA is required to run the model.")
        sys.exit(0)

    parser = setup_arg_parser()
    args = parser.parse_args()

    main(args)
