from pathlib import Path
from typing import Optional, Tuple, Dict
import json

import torch
import torch.nn as nn

from transformers import PreTrainedTokenizer, AutoTokenizer, AutoConfig, AutoModelForCausalLM, PreTrainedModel, logging
import accelerate

from utils import get_layer_mode_from_name, LayerMode, get_packed_info, get_model_path, find_layers, apply_dtype_to

import time
from transformers import logging

STRATEGY_FILE_NAME = "quant_strategy.json"

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


def apply_quant_strategy(name_attr, quant_strategy):
    """
    Apply quantization strategy based on the layer's name and the provided strategy.
    """
    strategy = None
    for key in ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']:
        if key in name_attr:
            strategy = quant_strategy[key]
            break
    return strategy


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
        name_attr = f'{name}.{attr}' if name else attr
        if name_attr in names:
            strategy = apply_quant_strategy(name_attr, quant_strategy) if quant_strategy else None
            if strategy:
                groups, rows = get_packed_info(tmp.in_features, strategy["bits"], strategy["bits_prop"], strategy["group_size"])
                bits = strategy["bits"][0]
                group_size = strategy["group_size"][str(bits)]

            common_params = {
                "in_channels": tmp.in_features, "out_channels": tmp.out_features,
                "w_bit": bits, "dtype": dtype, "group_size": group_size, "dq_group_size": 32,
                "use_gba_quant": True, "asym": False, "dq_mode": 2, "requires_grad": True
            }

            if layer_mode == LayerMode.LEGENCY:
                quantized_layer = MPQLinearCuda(**common_params)
            elif layer_mode == LayerMode.LAYER_MIX:
                quantized_layer = MBWQLinearCuda(use_mbw=False, **common_params)
            elif layer_mode == LayerMode.CHANNEL_MIX:
                quantized_layer = MBWQLinearCuda(use_mbw=True, groups=groups, rows_packed=rows, **common_params)
            else:
                raise NotImplementedError(f'Error: layer_mode: {layer_mode} not implemented yet.')

            setattr(module, attr, quantized_layer)

    for name_attr, child in module.named_children():
        name_sub = f'{name}.{name_attr}' if name else name_attr
        make_quant(child, names, layer_mode, name_sub, group_size, bits, dtype, quant_strategy)


def load_model(model_path: Path, layer_mode: LayerMode, bits: Optional[int] = None, group_size: Optional[int] = None,
               dtype: torch.dtype = torch.half, device_map: set = {"cuda:0"}, seqlen: int = 2048,
               model_config: Dict = {}) -> Tuple[nn.Module, AutoConfig]:
    """
    Load and initialize a model from the given path with options for quantization and device distribution.

    This function loads a model, applies quantization based on the provided layer mode, bits, and group size,
    and distributes the model across specified devices. It returns both the model and its configuration.

    Args:
        model_path (Path): Path to the model.
        layer_mode (LayerMode): Layer mode for quantization.
        bits (Optional[int]): Number of bits for quantization.
        group_size (Optional[int]): Group size for quantization.
        dtype (torch.dtype): Data type for model tensors.
        device_map (set): Devices to distribute the model.
        seqlen (int): Sequence length for the model.

    Returns:
        Tuple[nn.Module, AutoConfig]: A tuple containing the loaded and initialized model and its configuration.
    """
    start_time = time.time()
    print(Style.BRIGHT + Fore.CYAN + "Info: Loading Model ...")

    # Load quantization strategy if applicable
    strategy = {}
    if layer_mode in (LayerMode.LAYER_MIX, LayerMode.CHANNEL_MIX):
        strategy_path = model_path / STRATEGY_FILE_NAME
        try:
            with open(strategy_path, "r") as file:
                strategy = json.load(file)["measurement"]
        except FileNotFoundError:
            raise FileNotFoundError(f"Error: Strategy config file not found in {model_path}")

    # Initialize model with empty weights and load configuration
    with accelerate.init_empty_weights():
        config = AutoConfig.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(model_path, config=config, torch_dtype=dtype, **model_config).eval()

        # Quantize layers as necessary
        for i, layer in enumerate(model.model.layers):
            layers = find_layers(layer)
            layers_to_quantize = {name: layer for name, layer in layers.items() if name != 'lm_head'}

            strategy_per_block = strategy.get(f"model.layers.{i}") if strategy else None

            make_quant(layer, layers_to_quantize, layer_mode=layer_mode, group_size=group_size, bits=bits, dtype=dtype,
                       quant_strategy=strategy_per_block)

    # Load checkpoint, dispatch model, and apply post-initialization configurations
    model = accelerate.load_checkpoint_and_dispatch(model=model, checkpoint=model_path, device_map=device_map,
                                                    no_split_module_classes=["LlamaDecoderLayer"])
    model.seqlen = seqlen
    engine_layer_prepare(model)  # Assuming this prepares the model's engine layers post-initialization
    apply_dtype_to(model, dtype)  # Assuming this function applies the dtype to all model parameters

    print(Style.BRIGHT + Fore.GREEN + f"Info: Apply dtype: {dtype} to the model.")
    print(Style.BRIGHT + Fore.YELLOW + f'Total {torch.cuda.memory_allocated() / 1024**3:.2f} GiB VRAM used.')
    print(Style.BRIGHT + Fore.GREEN + f"Info: Loaded the model in {time.time() - start_time:.2f} seconds.")

    return model, config


def load(
    path_or_hf_repo: str,
    tokenizer_config: Dict = {},
    dtype: torch.dtype = torch.half,
    device_map: set = {"cuda:0"},
    seqlen: int = 2048,
    model_config: Dict = {},
) -> Tuple[nn.Module, PreTrainedTokenizer]:
    """
    Load the model and tokenizer from a given path or a huggingface repository.

    Args:
        path_or_hf_repo (str): The path or the huggingface repository to load the model from.
        tokenizer_config (dict, optional): Configuration parameters specifically for the tokenizer.
                                           Defaults to an empty dictionary.
        dtype (torch.dtype): Data type to be used.
        device_map (set): Device map defines how to distribute model on the GPUs.
        seqlen (int): Sequence length.
        adapter_file (str, optional): Path to the adapter file. If provided, applies LoRA layers to the model.
                                      Defaults to None.

    Returns:
        Tuple[nn.Module, PreTrainedTokenizer, AutoConfig]: A tuple containing the loaded model and tokenizer.

    Raises:
        FileNotFoundError: If the config file or safetensors are not found.
        ValueError: If the model class or args class are not found.
        NotImplementedError: If the LoRA layers functionality is not implemented.
    """
    if not ENGINE_AVAILABLE:
        raise Exception("Error: Bitorch Engine layers are not available.")

    layer_mode, bits, group_size = get_layer_mode_from_name(path_or_hf_repo)
    model_path = get_model_path(path_or_hf_repo)

    model, config = load_model(model_path, layer_mode, bits, group_size, dtype, device_map, seqlen, model_config)

    tokenizer = AutoTokenizer.from_pretrained(path_or_hf_repo, **tokenizer_config)

    return model, tokenizer, config

    generate(
        model,
        tokenizer,
        prompt,
        args.temp,
        args.max_tokens,
        True,
        top_p=args.top_p,
    )


def generate(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    prompt: str,
    temp: float = 0.8,
    max_tokens: int = 100,
    verbose: bool = False,
    top_p: float = 0.95,
    repetition_penalty: float = 1.5,
    top_k: int = 20,
    output_scores: bool = False,
    return_dict_in_generate: bool = True,
    output_attentions: bool = False,
    output_hidden_states: bool = False
) -> str:
    """
    Generates text from a given model and tokenizer with customizable generation settings.

    Args:
        model (PreTrainedModel): The pre-trained model from Hugging Face's Transformers.
        tokenizer (PreTrainedTokenizer): The tokenizer corresponding to the model.
        prompt (str): The initial text to begin generation.
        temp (float): The temperature for controlling creativity. Higher values generate more varied text.
        max_tokens (int): The maximum number of tokens to generate.
        verbose (bool): If True, additional information such as generation speed will be printed.
        top_p (float): The nucleus (top-p) sampling probability threshold for filtering tokens.
        repetition_penalty (float): The penalty for token repetition. Values >1 discourage repetition.
        top_k (int): The top-k sampling value. Limits generation to the top k probable tokens.
        output_scores (bool): Whether to return the generation scores.
        return_dict_in_generate (bool): Whether to return a dictionary in the `.generate()` method.
        output_attentions (bool): Whether to return attentions weights.
        output_hidden_states (bool): Whether to return hidden states.

    Returns:
        str: The generated text.
    """
    device = torch.device("cuda")
    model = model.to(device)
    model.eval()

    # Encode the prompt text to tensor
    input_ids = tokenizer.encode(prompt, return_tensors="pt", add_special_tokens=False).to(device)

    # Generation settings
    eos_token_id = tokenizer.eos_token_id
    prompt_length = len(input_ids[0])

    if verbose:
        logging.set_verbosity_info()
        print("=" * 50)
        print(f"Prompt: {prompt}")
        start_time = time.time()
    else:
        logging.set_verbosity_error()

    output_sequences = model.generate(
        input_ids=input_ids,
        temperature=temp,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        eos_token_id=eos_token_id,
        pad_token_id=eos_token_id,
        use_cache=True,
        max_new_tokens=max_tokens,
        top_k=top_k,
        output_scores=output_scores,
        return_dict_in_generate=return_dict_in_generate,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states
    )

    # Decode generated sequence
    generated_sequence = tokenizer.decode(output_sequences[0], clean_up_tokenization_spaces=True)

    # Remove the prompt at the beginning of the sequence
    generated_sequence = generated_sequence[len(tokenizer.decode(input_ids[0], clean_up_tokenization_spaces=True)):]

    if verbose:
        end_time = time.time()
        duration = end_time - start_time
        tokens_per_second = (prompt_length + max_tokens) / duration
        print(f"Generated text: {generated_sequence}")
        print(f"Duration: {duration:.2f}s, Tokens per second: {tokens_per_second:.2f}")

    return generated_sequence