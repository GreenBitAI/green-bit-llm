from pathlib import Path
from typing import Optional, Tuple
import json

import torch
import torch.nn as nn

from transformers import PreTrainedTokenizer, AutoTokenizer
import accelerate

from utils import get_layer_mode_from_name, LayerMode, get_packed_info, get_model_path, find_layers, apply_dtype_to
import time

strategy_fn = "quant_strategy.json"

ENGINE_AVAILABLE = True
try:
    from bitorch_engine.layers.qlinear.nbit import MPQLinearBase
    from bitorch_engine.layers.qlinear.nbit.cuda import MPQLinearCuda, MBWQLinearCuda
    from bitorch_engine.layers.qlinear.nbit.mps import MPQLinearMlx
except ModuleNotFoundError as e:
    ENGINE_AVAILABLE = False
    print(f"Error: Module not found: {e}.")

from colorama import init, Fore, Style
init(autoreset=True)


def engine_layer_prepare(model: torch.nn.Module):
    """
    Define the function that prepares the engine layer.

    Args:
        model (nn.Module): The PyTorch module will be operated.
    """
    target_layer_name = ''
    for n, m in model.named_modules():
        if issubclass(type(m), MPQLinearBase):
            m.prepare_params()
            if target_layer_name == '': target_layer_name = m.__class__.__name__
    print(Style.BRIGHT + Fore.GREEN + '{} parameter preparation finished.'.format(target_layer_name))


def make_quant(module, names, layer_mode: LayerMode, name='', group_size: Optional[int] = None, bits: Optional[int] = None,
               dtype: torch.dtype = torch.half, quant_strategy=None):
    """
    Applies quantization to the specified layers of a PyTorch module according to the provided quantization strategy.

    Args:
        module: The PyTorch module whose layers are to be quantized.
        names: A list of layer names within the module that need to be quantized.
        layer_mode: An enum (LayerMode) that specifies the quantization mode to be used.
        name: A base name to prefix to attributes for unique identification (used in recursive calls).
        group_size: The group size for quantization, affecting how weights are grouped during quantization.
        bits: The number of bits to use for weight quantization.
        dtype: The data type to be used for the quantized weights.
        quant_strategy: A dictionary defining specific quantization strategies for different layers.

    This function modifies the module in-place by replacing specified layers with their quantized counterparts.
    It supports different quantization modes and strategies, allowing fine-grained control over the quantization process.
    """

    if ENGINE_AVAILABLE and issubclass(type(module), MPQLinearBase):
        return

    for attr in dir(module):
        tmp = getattr(module, attr)
        name_attr = name + '.' + attr if name != '' else attr
        if name_attr in names:
            if layer_mode == LayerMode.LEGENCY:
                setattr(
                    module, attr, MPQLinearCuda(in_channels=tmp.in_features, out_channels=tmp.out_features,
                                w_bit=bits, dtype=dtype, group_size=group_size, dq_group_size=32,
                                use_gba_quant=True, asym=False, dq_mode=2, requires_grad=False)
                )
            else:
                groups = 64
                rows = 64
                if quant_strategy is not None:
                    for key in ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']:
                        if key in name_attr:
                            strategy = quant_strategy[key]
                            break

                    groups, rows = get_packed_info(tmp.in_features, strategy["bits"], strategy["bits_prop"],
                                                   strategy["group_size"])
                    bits = strategy["bits"][0]
                    group_size = strategy["group_size"][str(bits)]

                if layer_mode == LayerMode.LAYER_MIX:
                    setattr(
                        module, attr, MBWQLinearCuda(in_channels=tmp.in_features, out_channels=tmp.out_features,
                                                     w_bit=bits, dtype=dtype, group_size=group_size, dq_group_size=32, use_gba_quant=True,
                                                     asym=False, dq_mode=2, requires_grad=False, use_mbw=False)
                    )
                elif layer_mode == LayerMode.CHANNEL_MIX:
                    setattr(
                        module, attr, MBWQLinearCuda(in_channels=tmp.in_features, out_channels=tmp.out_features,
                                                     w_bit=bits, dtype=dtype, group_size=group_size, dq_group_size=32,
                                                     use_gba_quant=True, asym=False, dq_mode=2, requires_grad=False,
                                                     use_mbw=True, groups=groups, rows_packed=rows)
                    )
                else:
                    raise NotImplementedError('Error: layer_mode: {} not implemented yet.'.format(layer_mode))

    for name_attr, child in module.named_children():
        name_sub = name + '.' + name_attr if name != '' else name_attr
        make_quant(child, names, layer_mode, name_sub, group_size=group_size, bits=bits, dtype=dtype, quant_strategy=quant_strategy)


def load_model(model_path: Path, layer_mode: LayerMode, bits: Optional[int]=None, group_size: Optional[int]=None,
               dtype: torch.dtype = torch.half, device_map={"cuda:0"}, seqlen: int = 2048) -> nn.Module:
    """
    Load and initialize the model from a given path.

    Args:
        model_path (Path): The path to load the model from.
        layer_mode (LayerMode): for choosing which engine layer should be used.
        bits (int): bits for quantization
        group_size (int): group size used in quantization
        dtype (torch.dtype): data type to be used
        device_map (int): device map defines how to distribute model on the gpus.
        seqlen (int): sequence length

    Returns:
        nn.Module: The loaded and initialized model.

    Raises:
        FileNotFoundError: If the weight files (.safetensors) are not found.
        ValueError: If the model class or args class are not found or cannot be instantiated.
    """
    print(Style.BRIGHT + Fore.CYAN + "Info: Loading Model ...")
    t0 = time.time()

    if layer_mode in (LayerMode.LAYER_MIX, LayerMode.CHANNEL_MIX):
        try:
            with open(model_path / strategy_fn, "r") as file:
                strategy = json.load(file)["measurement"]
        except FileNotFoundError:
            raise Exception(f"Error: Strategy config file not found in {model_path}")

    with accelerate.init_empty_weights():
        config = AutoConfig.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(model_path, config=config, torch_dtype=dtype, trust_remote_code=True)
        model = model.eval()

        for i, layer in enumerate(model.model.layers):
            layers = find_layers(layer)

            strategy_per_block = None
            if layer_mode in (LayerMode.LAYER_MIX, LayerMode.CHANNEL_MIX):
                strategy_per_block = strategy["model.layers.{}".format(i)]

            for name in ['lm_head']:
                if name in layers:
                    del layers[name]

                make_quant(layer, layers, layer_mode=layer_mode, groupsize=group_size, bits=bits, dtype=dtype,
                           quant_strategy=strategy_per_block)

    model = accelerate.load_checkpoint_and_dispatch(
        model=model,
        checkpoint=model_path,
        device_map=device_map,
        no_split_module_classes=["LlamaDecoderLayer"]
    )
    model.seqlen = seqlen

    # post model.init() preparation
    engine_layer_prepare(model)
    # use dtype
    apply_dtype_to(model, dtype)
    print(Style.BRIGHT + Fore.GREEN + f"Info: Apply dtype: {str(dtype)} to the model.")

    print(Style.BRIGHT + Fore.YELLOW + 'Total {:.2f} Gib VRAM used.'.format(
        torch.cuda.memory_allocated() / 1024 / 1024 / 1024))

    print(Style.BRIGHT + Fore.GREEN + f"Info: Loaded the model in {(time.time() - t0):.2f} seconds.")

    return model, config


def load(
    path_or_hf_repo: str,
    tokenizer_config={},
    dtype: torch.dtype = torch.half,
    device_map = {"cuda:0"},
    seqlen: int = 2048,
    adapter_file: Optional[str] = None
) -> Tuple[nn.Module, PreTrainedTokenizer]:
    """
    Load the model and tokenizer from a given path or a huggingface repository.

    Args:
        path_or_hf_repo (Path): The path or the huggingface repository to load the model from.
        tokenizer_config (dict, optional): Configuration parameters specifically for the tokenizer. Defaults to an empty dictionary.
        dtype (torch.dtype): data type to be used
        device_map (int): device map defines how to distribute model on the gpus.
        seqlen (int): sequence length
        adapter_file (str, optional): Path to the adapter file. If provided, applies LoRA layers to the model.
            Defaults to None.
    Returns:
        Tuple[nn.Module, PreTrainedTokenizer]: A tuple containing the loaded model and tokenizer.

    Raises:
        FileNotFoundError: If config file or safetensors are not found.
        ValueError: If model class or args class are not found.
    """
    # check if Bitorch Engine avaliable
    if not ENGINE_AVAILABLE:
        raise Exception(f"Error: Bitorch Engine layers will not be available.")

    # parse model name pattern
    layer_mode, bits, group_size = get_layer_mode_from_name(path_or_hf_repo)
    model_path = get_model_path(path_or_hf_repo)

    model, config = load_model(model_path, layer_mode, bits, group_size, dtype, device_map, seqlen)

    tokenizer = AutoTokenizer.from_pretrained(model_path, **tokenizer_config)

    if adapter_file is not None:
        raise NotImplementedError('Error: LoRA layers not implemented yet.')

    return model, tokenizer, config