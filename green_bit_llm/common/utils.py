import re
from pathlib import Path
from typing import Dict

import torch.nn as nn
import torch

from huggingface_hub import snapshot_download
from transformers import AutoConfig

from auto_gptq.nn_modules.qlinear.qlinear_cuda import QuantLinear

from green_bit_llm.common.enum import LayerMode
from green_bit_llm.patches.deepseek_v3_moe_patch import detect_deepseek_v3_moe_model

STRATEGY_FILE_NAME = "quant_strategy.json"
MODEL_TYPE_QWEN2 = "qwen2"
STRATEGY_FILE_JSON_ROOT = "measurement"
GPTQ_QUANTIZATION_CONFIG = "quantization_config"


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
        return False
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

def check_quantization_config(config: AutoConfig):
    """
    Checks the quantization configuration of the given model configuration.

    Args:
        config (AutoConfig): The model configuration object.

    Returns:
        tuple: A tuple containing the layer mode, weight bits, and group size if the quantization configuration
               meets the specified criteria (quant_method is 'gptq' and sym is True). If the criteria are not met,
               returns (None, None, None).
    """
    if hasattr(config, GPTQ_QUANTIZATION_CONFIG):
        quant_config = config.quantization_config
        quant_config['disable_exllama'] = True
        # check if quant_method is gptq and sym must be True
        if quant_config.get('quant_method') == 'gptq' and quant_config.get('sym') == True:
            bits = quant_config.get('bits')
            group_size = quant_config.get('group_size')
            return LayerMode.LEGENCY, bits, group_size

    return None, None, None

def get_layer_mode(model_name: str, config: AutoConfig) -> tuple:
    """
    Determines the layer mode from the model's config or from the model name and extracts weight bits and group size if present.

    Args:
        model_name (str): The name of the model.
        config (AutoConfig): The model configuration object.

    Returns:
        tuple: A tuple containing the layer mode, weight bits, and group size. If the model's quantization
               configuration does not meet the specified criteria, it matches the model name pattern.

    Raises:
        ValueError: If the model name is invalid or unsupported.
    """
    mode, bits, group_size = check_quantization_config(config)
    if bits is not None:
        return mode, bits, group_size

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

def find_layers(module: nn.Module, layers=[nn.Conv2d, nn.Linear, QuantLinear], name=''):
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
    if model.dtype is dtype:
        return

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

def apply_quant_strategy(name_attr: str, quant_strategy: Dict):
    """
    Apply quantization strategy based on the layer's name and the provided strategy.
    Updated to support DeepSeek V2 MoE models and other complex architectures.
    """
    strategy = None

    # DeepSeek V2 style attention projections (decomposed Q/K/V)
    deepseek_attention_mapping = {
        'q_a_proj': 'q_a_proj',
        'q_b_proj': 'q_b_proj',
        'kv_a_proj_with_mqa': 'kv_a_proj_with_mqa',
        'kv_b_proj': 'kv_b_proj'
    }

    for layer_name, strategy_key in deepseek_attention_mapping.items():
        if layer_name in name_attr:
            try:
                strategy = quant_strategy[strategy_key]
                return strategy
            except KeyError:
                pass

    # Standard attention projections (for backward compatibility)
    standard_attention_keys = ['q_proj', 'k_proj', 'v_proj', 'o_proj']
    for key in standard_attention_keys:
        if key in name_attr and not any(prefix in name_attr for prefix in ['q_a_', 'q_b_', 'kv_a_', 'kv_b_']):
            try:
                strategy = quant_strategy[key]
                return strategy
            except KeyError:
                pass

    # MoE gate layer (router) - includes both weight and bias
    # DeepSeek V2 has: mlp.gate.weight and mlp.gate.e_score_correction_bias
    if ('mlp.gate.' in name_attr or name_attr.endswith('mlp.gate')) and 'experts' not in name_attr:
        try:
            strategy = quant_strategy['moe_gate']
            return strategy
        except KeyError:
            pass

    # MoE shared expert layers (DeepSeek V2 specific)
    # Note: actual path is 'mlp.shared_experts.' (plural)
    if 'mlp.shared_experts.' in name_attr or 'shared_experts.' in name_attr:
        if '.gate_proj' in name_attr or 'gate_proj' in name_attr:
            try:
                strategy = quant_strategy['moe_shared_expert_gate_proj']
                return strategy
            except KeyError:
                pass
        elif '.up_proj' in name_attr or 'up_proj' in name_attr:
            try:
                strategy = quant_strategy['moe_shared_expert_up_proj']
                return strategy
            except KeyError:
                pass
        elif '.down_proj' in name_attr or 'down_proj' in name_attr:
            try:
                strategy = quant_strategy['moe_shared_expert_down_proj']
                return strategy
            except KeyError:
                pass

    # MoE expert layers - match any expert number (supports 100+ experts)
    if 'mlp.experts.' in name_attr:
        if '.gate_proj' in name_attr:
            try:
                strategy = quant_strategy['moe_expert_gate_proj']
                return strategy
            except KeyError:
                pass
        elif '.up_proj' in name_attr:
            try:
                strategy = quant_strategy['moe_expert_up_proj']
                return strategy
            except KeyError:
                pass
        elif '.down_proj' in name_attr:
            try:
                strategy = quant_strategy['moe_expert_down_proj']
                return strategy
            except KeyError:
                pass

    # Fallback to standard FFN layers (non-MoE layers)
    standard_ffn_keys = ['gate_proj', 'up_proj', 'down_proj']
    for key in standard_ffn_keys:
        if key in name_attr and 'experts' not in name_attr and 'shared_experts' not in name_attr:
            try:
                strategy = quant_strategy[key]
                return strategy
            except KeyError:
                pass

    # Additional fallback for other layer types
    fallback_keys = ['qkv_proj', 'gate_up_proj']
    for key in fallback_keys:
        if key in name_attr:
            try:
                strategy = quant_strategy[key]
                return strategy
            except KeyError:
                pass

    return strategy

def detect_moe_model_type(config):
    """
    Detect MoE model type and return model information
    """
    model_info = {
        'type': 'standard',
        'needs_patch': False,
        'experts_count': 0,
        'has_shared_experts': False
    }

    # Check Qwen3 MoE
    if hasattr(config, 'model_type') and 'qwen3' in config.model_type.lower():
        if getattr(config, 'num_experts', 0) > 0:
            model_info.update({
                'type': 'qwen3_moe',
                'needs_patch': True,
                'experts_count': config.num_experts,
                'has_shared_experts': False
            })

    # Check DeepSeek V3 MoE
    elif detect_deepseek_v3_moe_model(config):
        model_info.update({
            'type': 'deepseek_v3_moe',
            'needs_patch': True,
            'experts_count': getattr(config, 'n_routed_experts', 0),
            'has_shared_experts': getattr(config, 'n_shared_experts', 0) > 0
        })

    return model_info