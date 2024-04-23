import argparse
import sys

import torch

from transformers import PreTrainedTokenizer, TrainingArguments
from datasets import load_dataset
from trl import SFTTrainer
from peft import PeftModel, LoraConfig, get_peft_model

from green_bit_llm import load, setup_shared_arg_parser
from .peft_utils.model import *
from .optim import AdamW8bit

import warnings
warnings.filterwarnings('ignore')

try:
    from bitorch_engine.optim import DiodeMix
except ModuleNotFoundError as e:
    raise Exception(f"Error occurred while importing Bitorch Engine module '{str(e)}'.")

from .utils import str_to_torch_dtype

# default value for arguments
DEFAULT_MODEL_PATH = "GreenBitAI/Qwen-1.5-1.8B-layer-mix-bpw-3.0"
DEFAULT_SEQLEN = 512
DEFAULT_RANDOM_SEED = 0
DEFAULT_LR = 1e-5
DEFAULT_LR_GALORE = 1e-4
DEFAULT_LR_FP = 1e-6
DEFAULT_BETAS = (0.9, 0.999)


def setup_arg_parser():
    """Set up and return the argument parser."""
    parser = setup_shared_arg_parser("green-bit-llm lora script")
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_RANDOM_SEED,
        help="The random seed for data loader.",
    )
    # qweight related
    parser.add_argument(
        "--lr-fp",
        type=float,
        default=DEFAULT_LR_FP,
        help="Learning rate for full-precision weight."
    )
    parser.add_argument("--lora-rank", type=int, default=64)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.01)
    return parser


def create_param_groups(model, args: argparse.ArgumentParser):
    """
    Create parameter groups for parameter efficient finetuning.
    Args:
        model (nn.Module): The neural network model.
        args (argparse.ArgumentParser): Command line arguments for additional parameters.

    Returns:
        List[dict]: A list of dictionaries where each dictionary contains a parameter group.
    """
    params_groups = []

    # Create list of peft parameters
    params_lora = [p for n, p in model.named_parameters() if "lora" in n]
     
    params_group_lora = {'params': params_lora, 'lr': args.lr_fp, 'betas': DEFAULT_BETAS}
    
    params_groups.append(params_group_lora)

    return params_groups


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
        requires_grad=False,
    )
    
    config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "v_proj", "out_proj", "down_proj", "up_proj"],
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    
    replace_peft_lora_model_with_gba_lora_model()

    model = get_peft_model(model, config)
    
    model.print_trainable_parameters()
     
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
    
    for name, param in model.named_parameters():
        if "qweight" not in name:
            param.requires_grad = True
    
    trainer = SFTTrainer(
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
        print("Warning: CUDA is needed to run the model.")
        sys.exit(0)

    parser = setup_arg_parser()
    args = parser.parse_args()

    main(args)
