import argparse
import sys

import torch
import torch.nn as nn

from transformers import PreTrainedTokenizer, TrainingArguments
from datasets import load_dataset
from trl import SFTTrainer
from peft import PeftModel, LoraConfig, get_peft_model

from green_bit_llm.model import load
from .peft_utils.model import *

import warnings

warnings.filterwarnings('ignore')

ENGINE_AVAILABLE = True
try:
    from bitorch_engine.optim import DiodeMix
except ModuleNotFoundError as e:
    ENGINE_AVAILABLE = False
    raise Exception("Error: Bitorch Engine optimizer (DiodeMix) are not available.")

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
    parser.add_argument(
        "--dataset",
        type=str,
        default="tatsu-lab/alpaca",
        help="Dataset name for finetuning",
    )
    # qweight related
    parser.add_argument(
        "--lr-fp",
        type=float,
        default=DEFAULT_LR_FP,
        help="Learning rate for full-precision weight."
    )
    parser.add_argument("--optimizer", default="DiodeMix")
    return parser


def str_to_torch_dtype(dtype: str):
    if dtype is None:
        return None
    elif dtype == "float":
        return torch.float
    elif dtype == "half":
        return torch.half
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")


def create_device_map(cuda_device_id):
    ids = cuda_device_id.split(',')
    # Create strings in the format "cuda:x" for each ID and put them into the collection
    device_map = {f"cuda:{id}" for id in ids}
    return device_map


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

    if args.eos_token is not None:
        tokenizer_config["eos_token"] = args.eos_token

    model, tokenizer, config = load(
        args.model,
        tokenizer_config=tokenizer_config,
        device_map='auto',
        seqlen=args.seqlen,
        model_config=pretrain_model_config,
        requires_grad=False,
    )
    
    config = LoraConfig(
        r=64,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj", "out_proj", "down_proj", "up_proj"],
        lora_dropout=0.01,
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
                    per_device_train_batch_size=4,
                    logging_steps=1,
                    save_steps=50,
                    max_grad_norm=0, # NOTE: max_grad_norm MUST be <= 0 or None, otherwise raise dtype error due to the Int dtype of qweight.
                )

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
