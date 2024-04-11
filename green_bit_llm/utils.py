from pathlib import Path
import re
import torch.nn as nn
import torch

from huggingface_hub import snapshot_download

from .enum import LayerMode

def check_engine_available()-> bool:
    """
    Checks if the required modules for Bitorch Engine are available.
    Returns True if the modules are found, False otherwise.
    """
    try:
        from bitorch_engine.layers.qlinear.nbit import MPQLinearBase
        from bitorch_engine.layers.qlinear.nbit.cuda import MPQLinearCuda, MBWQLinearCuda
    except ModuleNotFoundError as e:
        print(f"Error: Module not found: {e}.")
    return True


def match_model_pattern(model_name: str) -> tuple:
    """
    Matches the model name against known patterns and extracts relevant details.

    Args:
        model_name (str): The name of the model.

    Returns:
        tuple: A tuple containing the layer mode, weight bits, and group size.

    Raises:
        ValueError: If the model name does not match any known pattern.
    """
    if model_name.endswith('-mlx'):
        raise ValueError("Model name ends with '-mlx', which is not supported.")

    patterns = {
        r'channel-mix': LayerMode.CHANNEL_MIX,
        r'layer-mix': LayerMode.LAYER_MIX,
        r'B-(\d+)bit-groupsize(\d+)': LayerMode.LEGENCY,
        r'w(\d+)a\d+g(\d+)': LayerMode.LEGENCY,
    }

    for pattern, mode in patterns.items():
        match = re.search(pattern, model_name)
        if match:
            groups = match.groups()
            if mode == LayerMode.LEGENCY and groups:
                return mode, int(groups[0]), int(groups[1])
            return mode, None, None

    raise ValueError(f"Invalid or unsupported model name pattern: {model_name}")


def get_layer_mode_from_name(model_name: str) -> tuple:
    """
    Determines the layer mode from the model name and extracts weight bits and group size if present.

    Args:
        model_name (str): The name of the model.

    Returns:
        tuple: The layer mode, weight bits, and group size.

    Raises:
        ValueError: If the model name is invalid or unsupported.
    """
    return match_model_pattern(model_name)


def get_packed_info(channels, n_bits, bits_prop, bits_group_size):
    """
    Calculate the distribution of channels into groups and rows based on their bit properties.

    Args:
        channels (int): Total number of channels to be packed.
        n_bits (list of int): Number of bits per channel for each group.
        bits_prop (list of float): Proportion of channels to be allocated to each bit type.
        bits_group_size (dict): Mapping of group index to the minimal number of channels in a group.

    Returns:
        Tuple[int, int]: A tuple containing the total number of groups and rows needed.
    """
    groups = 0
    rows = 0
    bits_channel = []
    for idx in range(len(bits_prop)):
        if idx < len(bits_prop) - 1:
            minimal_channels = list(bits_group_size.values())[idx]
            channel_pre_pack = max(1, int(channels * (bits_prop[idx])) // minimal_channels) * minimal_channels
            bits_channel.append(channel_pre_pack)
            groups += channel_pre_pack // minimal_channels
            rows += channel_pre_pack // 32 * n_bits[idx]
        else:
            minimal_channels = list(bits_group_size.values())[idx]
            channel_pre_pack = channels - sum(bits_channel)
            bits_channel.append(channel_pre_pack)
            groups += channel_pre_pack // minimal_channels
            rows += channel_pre_pack // 32 * n_bits[idx]
    return groups, rows



def find_layers(module: nn.Module, layers=[nn.Conv2d, nn.Linear], name=''):
    """
    Recursively searches and returns a dictionary of layers within a given PyTorch model
    (or module) that match the specified types.

    Args:
        module (nn.Module): The model or module to search within.
        layers (list): A list of layer classes (e.g., nn.Conv2d, nn.Linear) to look for within the model.
                         Default is [nn.Conv2d, nn.Linear].
        name (str): The namespace (hierarchical path) leading to the current module being searched.
                      This is used internally for recursive calls to build the full path names of layers.

    Returns:
        dict: A dictionary where each key is the hierarchical name of the layer (showing its path within
                the model) and each value is the corresponding layer module itself.
    """
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_layers(
            child, layers=layers, name=name + '.' + name1 if name != '' else name1
        ))
    return res


def get_model_path(path_or_hf_repo: str, token=None) -> Path:
    """
    Ensures the model is available locally. If the path does not exist locally,
    it is downloaded from the Hugging Face Hub.

    Args:
        path_or_hf_repo (str): The local path or Hugging Face repository ID of the model.
        token (Optional): user token to access HF repo.

    Returns:
        Path: The path to the model.
    """
    model_path = Path(path_or_hf_repo)
    if not model_path.exists():
        model_path = Path(
            snapshot_download(
                repo_id=path_or_hf_repo,
                allow_patterns=[
                    "*.json",
                    "*.safetensors",
                    "*.py",
                    "tokenizer.model",
                    "*.tiktoken",
                    "*.txt",
                ],
                token=token
            )
        )
    return model_path


def apply_dtype_to(model: nn.Module, dtype: torch.dtype):
    """
    Applies a specified data type to the given PyTorch model. This function
    is useful for model conversion in mixed precision training, where different
    parts of the model may benefit from different precision levels for efficiency
    and speed.

    Args:
        model: The PyTorch model to modify.
        dtype: The target data type to apply to the model. Supported types are
                 torch.float, torch.half, and torch.bfloat16.
    Returns:
        None. The model is modified in-place.
    """

    # Check if the specified dtype is torch.half, and if so, convert the model to half precision.
    if dtype is torch.half:
        model.half()
    # Check if the specified dtype is torch.bfloat16, and if so, convert the model to bfloat16 precision.
    elif dtype is torch.bfloat16:
        model.bfloat16()
    # Additional condition for full precision (float32).
    elif dtype is torch.float:
        model.float()
    else:
        # If the dtype is not supported, raise an error.
        raise ValueError("Unsupported dtype specified. Supported dtypes are torch.float, torch.half, and torch.bfloat16.")