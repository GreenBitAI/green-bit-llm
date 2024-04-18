import argparse
import sys

import torch
import torch.nn as nn

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

    # GaLore parameters
    parser.add_argument(
        "--galore",
        action="store_true",
        help="Enable using galore",
    )
    parser.add_argument("--rank", type=int, default=128)
    parser.add_argument("--update_proj_gap", type=int, default=50)
    parser.add_argument("--galore_scale", type=float, default=1.0)
    parser.add_argument("--proj_type", type=str, default="std")
    return parser


def create_device_map(cuda_device_id):
    ids = cuda_device_id.split(',')
    # Create strings in the format "cuda:x" for each ID and put them into the collection
    device_map = {f"cuda:{id}" for id in ids}
    return device_map


def create_param_groups(model:nn.Module, args: argparse.ArgumentParser):
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

        params_group_2bit = {'params': params_2_bit, 'lr': 2e-3, 'betas': (0.9, 0.999)}
        params_group_4bit = {'params': params_4_bit, 'lr': 1e-3, 'betas': (0.9, 0.999)}
        params_group_regular = {'params': params_regular, 'lr': 1e-5, 'betas': (0.9, 0.999)}

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
            params_group_regular,
            params_group_2bit,
            params_group_4bit
        ]
        return param_groups


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

    train_args = TrainingArguments(output_dir="./output",
                            gradient_checkpointing=True,
                            auto_find_batch_size=True,
                            max_grad_norm=0 # NOTE: max_grad_norm MUST < 0 or None, otherwise raise dtype error due to the Int dtype of qweight.
                            )

    param_groups = create_param_groups(model, args)

    optimizer = DiodeMix(param_groups)

    optimizers = (optimizer, None)

    trainer = SFTTrainer(
        model=model,
        args=train_args,
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
