import sys
import os

import torch

from transformers import PreTrainedTokenizer, TrainingArguments
from datasets import load_dataset
from green_bit_llm.sft.trainer import GbaSFTTrainer

from green_bit_llm.common import load
from green_bit_llm.args_parser import setup_shared_arg_parser

import warnings

warnings.filterwarnings('ignore')

try:
    from bitorch_engine.optim import DiodeMix
    from bitorch_engine.layers.qlinear.nbit import MPQLinearBase
except ModuleNotFoundError as e:
    raise Exception(f"Error occurred while importing Bitorch Engine module '{str(e)}'.")

from green_bit_llm.sft.optim import AdamW8bit
from green_bit_llm.sft.utils import str_to_torch_dtype, create_param_groups


# default value for arguments
DEFAULT_MODEL_PATH = "GreenBitAI/Qwen-1.5-1.8B-layer-mix-bpw-3.0"
DEFAULT_SEQLEN = 512
DEFAULT_RANDOM_SEED = 0
DEFAULT_LR = 5e-6
DEFAULT_LR_GALORE = 5e-5
DEFAULT_LR_ADAMW8BIT = 5e-3
DEFAULT_BETAS = (0.9, 0.99)


def setup_arg_parser():
    """Set up and return the argument parser."""
    parser = setup_shared_arg_parser("green-bit-llm finetune script")
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_RANDOM_SEED,
        help="The random seed for data loader.",
    )
    # GaLore parameters
    parser.add_argument(
        "--galore",
        action="store_true",
        help="Enable using galore",
    )
    parser.add_argument("--galore-rank", type=int, default=256)
    parser.add_argument("--galore-update-proj-gap", type=int, default=200)
    parser.add_argument("--galore-scale", type=float, default=0.25)
    parser.add_argument("--galore-proj-type", type=str, default="std")

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
    return parser


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

    # NOTE:
    # Typically, Hugging Face's Trainer does not support fine-tuning quantized models.
    # However, our tool supports this scenario.
    # Therefore, we need to delete this attribute after loading the model.
    if hasattr(model, 'is_quantized'):
        delattr(model, 'is_quantized')

    param_groups = create_param_groups(model, args, DEFAULT_BETAS, DEFAULT_LR_GALORE, DEFAULT_LR_ADAMW8BIT, DEFAULT_LR)

    model.train()

    dataset = load_dataset(args.dataset, split="train")
    
    if not args.galore:
        args.save_dir = os.path.join(args.save_dir, "finetune/common/", args.model)
    else:
        args.save_dir = os.path.join(args.save_dir, "finetune/galore/", args.model)
    

    train_args = TrainingArguments(
                    output_dir=args.save_dir,
                    gradient_checkpointing=True,
                    #auto_find_batch_size=True,
                    per_device_train_batch_size=args.batch_size,
                    logging_steps=1,
                    num_train_epochs=1,
                    save_steps=args.save_step,
                    save_total_limit=3,
                    gradient_accumulation_steps=1,
                    lr_scheduler_type='cosine',
                    max_grad_norm=0, # NOTE: max_grad_norm MUST be <= 0 or None, otherwise raise dtype error due to the Int dtype of qweight.
                )

    # Optimizer
    if 'adamw8bit' in args.optimizer.lower():
        optimizer = AdamW8bit(param_groups, weight_decay=args.weight_decay, dtype=str_to_torch_dtype(args.dtype))
    elif 'diodemix' in args.optimizer.lower():
        optimizer = DiodeMix(param_groups, dtype=str_to_torch_dtype(args.dtype))

    optimizers = (optimizer, None)

    # Trainer
    trainer = GbaSFTTrainer(
        model=model,
        args=train_args,
        train_dataset=dataset,
        dataset_text_field="text",
        optimizers=optimizers,
        max_seq_length=args.seqlen,
    )

    trainer.train()

    model.save_pretrained(args.save_dir)


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("Warning: CUDA is required to run the model.")
        sys.exit(0)

    parser = setup_arg_parser()
    args = parser.parse_args()

    main(args)
