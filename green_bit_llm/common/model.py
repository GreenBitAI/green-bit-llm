import os
import json
import time
from pathlib import Path
from typing import Optional, Tuple, Dict

import torch
import torch.nn as nn

import accelerate
from transformers import (
    PreTrainedTokenizer,
    AutoTokenizer,
    AutoConfig,
    AutoModelForCausalLM,
    PreTrainedModel,
    logging
)

from auto_gptq.nn_modules.qlinear.qlinear_cuda import QuantLinear

from green_bit_llm.common.enum import LayerMode, TextGenMode
from green_bit_llm.common.utils import (
    get_layer_mode,
    get_packed_info,
    get_model_path,
    find_layers,
    apply_dtype_to,
    apply_quant_strategy,
    detect_moe_model_type
)

from green_bit_llm.common.utils import (
    STRATEGY_FILE_NAME,
    MODEL_TYPE_QWEN2,
    STRATEGY_FILE_JSON_ROOT
)
from green_bit_llm.patches.qwen3_moe_patch import apply_qwen3_moe_patch, restore_qwen3_moe_patch
from green_bit_llm.patches.deepseek_v3_moe_patch import (
    apply_deepseek_v3_moe_patch,
    restore_deepseek_v3_moe_patch
)

from colorama import init, Fore, Style
init(autoreset=True)


ENGINE_AVAILABLE = True
try:
    from bitorch_engine.layers.qlinear.nbit import MPQLinearBase
    from bitorch_engine.layers.qlinear.nbit.cuda import MPQLinearCuda, MBWQLinearCuda
except ModuleNotFoundError as e:
    ENGINE_AVAILABLE = False
    print(f"Error: Module not found: {e}.")


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
    print(Style.BRIGHT + Fore.CYAN + 'Info: {} parameter preparation finished.'.format(target_layer_name))

def get_disable_bias(name_attr: str, model_type) -> bool:
    """
    Usually the linear layer of llm disables bias. The three components 'q_proj', 'k_proj', and 'v_proj' in the qwen2 model attention module are exceptions.
    Updated to handle DeepSeek V3 decomposed attention layers.
    """
    # Original Qwen2 exception
    for key in ['q_proj', 'k_proj', 'v_proj']:
        if key in name_attr and model_type == MODEL_TYPE_QWEN2:
            return False

    # DeepSeek V3 decomposed attention layers might have different bias settings
    # Check for decomposed attention layers
    deepseek_attention_layers = ['q_a_proj', 'q_b_proj', 'kv_a_proj_with_mqa', 'kv_b_proj']
    for key in deepseek_attention_layers:
        if key in name_attr:
            # For DeepSeek models, these layers typically don't use bias
            return True

    return True

def make_quant(module, names, layer_mode: LayerMode, name='', group_size: Optional[int] = None, bits: Optional[int] = None,
               dtype: torch.dtype = torch.half, quant_strategy = None, model_type = None, requires_grad: bool = False):
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
        model_type: This attribute information has been read from config.json. E.g. "qwen2" for Qwen-model
        requires_grad (bool): Set if the model requires gradient. It can be set to True for training.

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
                # Handle special case where moe_gate might not be quantized
                if isinstance(strategy, dict) and 'desc' in strategy and 'Not quantized' in strategy['desc']:
                    print(Style.BRIGHT + Fore.YELLOW + f"Info: Skipping quantization for {name_attr} (marked as not quantized)")
                    continue

                groups, rows = get_packed_info(tmp.in_features, strategy["bits"], strategy["bits_prop"], strategy["group_size"])
                bits = strategy["bits"][0]
                group_size = strategy["group_size"][str(bits)]

            disable_bias = get_disable_bias(name_attr, model_type)

            # Check if auto_gptq.QuantLinear is used
            is_auto_gptq_quant_linear = isinstance(tmp, QuantLinear)
            use_gba_quant = not is_auto_gptq_quant_linear
            asym = is_auto_gptq_quant_linear
            in_c = tmp.infeatures if is_auto_gptq_quant_linear else tmp.in_features
            out_c = tmp.outfeatures if is_auto_gptq_quant_linear else tmp.out_features

            common_params = {
                "in_channels": in_c,
                "out_channels": out_c,
                "w_bit": bits, "dtype": dtype, "group_size": group_size, "dq_group_size": 32,
                "use_gba_quant": use_gba_quant, "asym": asym, "dq_mode": 2, "requires_grad": requires_grad,
                "disable_bias": disable_bias
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
        make_quant(child, names, layer_mode, name_sub, group_size, bits, dtype, quant_strategy, model_type, requires_grad)

def load_strategy_file(model_path: Path, layer_mode: LayerMode) -> Dict:
    """
        Load strategy configuration from a file based on the provided layer mode.

        Args:
            model_path (Path): The file system path to the model directory.
            layer_mode (LayerMode): The mode of the layers, either 'layer_mix' or 'channel_mix'.

        Returns:
            Dict: A dictionary containing the strategy configuration if the layer_mode is
                  'layer_mix' or 'channel_mix'. Returns an empty dictionary otherwise.

        Raises:
            FileNotFoundError: If the strategy configuration file is not found in the specified path.
    """
    strategy = {}
    if layer_mode in (LayerMode.LAYER_MIX, LayerMode.CHANNEL_MIX):
        strategy_path = os.path.join(model_path, STRATEGY_FILE_NAME)
        try:
            with open(strategy_path, "r") as file:
                strategy = json.load(file)[STRATEGY_FILE_JSON_ROOT]
        except FileNotFoundError:
            raise FileNotFoundError(f"Error: Strategy config file not found in {model_path}")
    return strategy

def load_model(model_path: Path, config: AutoConfig, layer_mode: LayerMode, bits: Optional[int] = None,
               group_size: Optional[int] = None, dtype: torch.dtype = torch.half, device_map: set = {"cuda:0"},
               seqlen: int = 2048, model_config: Dict = {}, requires_grad: bool = False) -> nn.Module:
    """
    Load and initialize a model from the given path with options for quantization and device distribution.

    This function loads a model, applies quantization based on the provided layer mode, bits, and group size,
    and distributes the model across specified devices. It returns both the model and its configuration.

    Args:
        model_path (Path): Path to the model.
        config (Dict): configs loaded from hf-model-config.
        layer_mode (LayerMode): Layer mode for quantization.
        bits (Optional[int]): Number of bits for quantization.
        group_size (Optional[int]): Group size for quantization.
        dtype (torch.dtype): Data type for model tensors.
        device_map (set): Devices to distribute the model.
        seqlen (int): Sequence length for the model.
        model_config (str, optional): Model configurations for "AutoModelForCausalLM.from_pretrained".
        requires_grad (bool): Set if the model requires gradient. It can be set to True for training.

    Returns:
        nn.Module: The loaded and initialized model.
    """
    logging.set_verbosity_error()

    # Detect MoE model type and apply appropriate patches
    moe_info = detect_moe_model_type(config)
    applied_patches = []

    if moe_info['needs_patch']:
        print(Style.BRIGHT + Fore.CYAN + f"Info: Detected {moe_info['type'].upper()} model")
        print(f"   Experts: {moe_info['experts_count']}, Shared experts: {moe_info['has_shared_experts']}")
        print("   Applying quantization-friendly patch...")

        try:
            if moe_info['type'] == 'qwen3_moe':
                apply_qwen3_moe_patch()
                applied_patches.append('qwen3_moe')

            elif moe_info['type'] == 'deepseek_v3_moe':
                # Use hybrid strategy for DeepSeek V3 due to large number of experts
                strategy = 'hybrid' if moe_info['experts_count'] > 50 else 'conservative'
                apply_deepseek_v3_moe_patch(strategy)
                applied_patches.append('deepseek_v3_moe')

        except Exception as e:
            print(Style.BRIGHT + Fore.YELLOW + f"Warning: Patch application failed: {e}")
            print("   Using original implementation, may have compatibility issues")

    try:
        start_time = time.time()
        print(Style.BRIGHT + Fore.CYAN + "Info: Loading Model ...")
        print(f"Model path: {model_path}")
        print(f"Layer mode: {layer_mode}")

        # Load quantization strategy if applicable
        strategy = load_strategy_file(model_path, layer_mode)

        # Initialize model with empty weights and load configuration
        with accelerate.init_empty_weights():
            model = AutoModelForCausalLM.from_pretrained(model_path, config=config, torch_dtype=dtype, **model_config).eval()

            # Check if this is a large MoE model and warn user about processing time
            total_layers = len(model.model.layers)

            if moe_info['needs_patch']:
                processing_note = "This may take longer to process"
                if moe_info['type'] == 'deepseek_v3_moe':
                    processing_note += f" (256 experts)"
                elif moe_info['type'] == 'qwen3_moe':
                    processing_note += f" ({moe_info['experts_count']} experts per layer)"

                print(Style.BRIGHT + Fore.CYAN + f"Info: {processing_note}...")

            # Quantize layers as necessary
            for i, layer in enumerate(model.model.layers):
                layers = find_layers(layer)

                # Filter out lm_head and prepare layers for quantization
                layers_to_quantize = {name: layer for name, layer in layers.items() if name != 'lm_head'}

                # Get strategy for this specific layer block
                strategy_per_block = strategy.get(f"model.layers.{i}") if strategy else None

                # Enhanced logging for MoE layers
                if strategy_per_block and i < 5:  # Only log first few layers to avoid spam
                    layer_type = "Standard FFN"

                    if moe_info['type'] == 'qwen3_moe' and 'moe_expert_gate_proj' in strategy_per_block:
                        expert_count = sum(1 for k in layers_to_quantize.keys() if 'experts.' in k and 'gate_proj' in k)
                        layer_type = f"Qwen3 MoE with {expert_count // 3} experts"
                        if 'moe_shared_expert_gate_proj' in strategy_per_block:
                            layer_type += " + Shared Expert"

                    elif moe_info['type'] == 'deepseek_v3_moe':
                        if any('experts.' in k for k in layers_to_quantize.keys()):
                            expert_count = sum(
                                1 for k in layers_to_quantize.keys() if 'experts.' in k and 'gate_proj' in k)
                            layer_type = f"DeepSeek V3 MoE with {expert_count // 3} routed experts"
                            if any('shared_experts.' in k for k in layers_to_quantize.keys()):
                                layer_type += " + shared experts"

                    print(Style.BRIGHT + Fore.CYAN + f"Info: Layer {i}: {layer_type}")

                make_quant(layer, layers_to_quantize, layer_mode=layer_mode, group_size=group_size, bits=bits, dtype=dtype,
                           quant_strategy=strategy_per_block, model_type=config.model_type, requires_grad=requires_grad)

                # Progress reporting - more frequent for large MoE models
                progress_interval = 2 if moe_info['experts_count'] > 100 else 5
                if (i + 1) % progress_interval == 0:
                    progress_pct = (i + 1) / total_layers * 100
                    print(
                        Style.BRIGHT + Fore.CYAN + f"Info: Quantized {i + 1}/{total_layers} layers ({progress_pct:.1f}%)")

        # If we need to bind weights, delete the meta tensor of lm_head
        if config.tie_word_embeddings:
            print(Style.BRIGHT + Fore.CYAN + "Info: Using tied word embeddings, removing lm_head meta tensor")
            if hasattr(model, 'lm_head'):
                if hasattr(model.lm_head, 'weight'):
                    delattr(model.lm_head, 'weight')
            model.tie_weights()

        # Load checkpoint, dispatch model, and apply post-initialization configurations
        # Extended no_split_module_classes for various MoE architectures
        no_split_classes = [
            "LlamaDecoderLayer", "MixtralDecoderLayer", "Qwen2MoeDecoderLayer",
            "Qwen3MoeDecoderLayer", "DeepSeekV3DecoderLayer"
        ]

        print(Style.BRIGHT + Fore.CYAN + "Info: Loading model weights and dispatching to devices...")
        # Load checkpoint, dispatch model, and apply post-initialization configurations
        model = accelerate.load_checkpoint_and_dispatch(model=model, checkpoint=model_path, device_map=device_map,
                                                        no_split_module_classes=no_split_classes)
        model.seqlen = seqlen

        print(Style.BRIGHT + Fore.CYAN + "Info: Preparing engine layers...")
        engine_layer_prepare(model)  # Assuming this prepares the model's engine layers post-initialization

        print(Style.BRIGHT + Fore.CYAN + "Info: Applying final dtype conversion...")
        apply_dtype_to(model, dtype)  # Assuming this function applies the dtype to all model parameters

        print(Style.BRIGHT + Fore.CYAN + f"Info: Apply dtype: {dtype} to the model.")
        print(Style.BRIGHT + Fore.CYAN + f'Info: Total {torch.cuda.memory_allocated() / 1024**3:.2f} GiB VRAM used.')
        print(Style.BRIGHT + Fore.CYAN + f"Info: Loaded the model in {time.time() - start_time:.2f} seconds.")

        # Success message for applied patches
        if applied_patches:
            print(Style.BRIGHT + Fore.GREEN + f"Info: Successfully applied patches: {', '.join(applied_patches)}")

        return model
    except Exception as e:
        # If error occurs, restore all patches
        print(Style.BRIGHT + Fore.RED + f"Error: Model loading failed: {e}")

        if 'qwen3_moe' in applied_patches:
            restore_qwen3_moe_patch()
        if 'deepseek_v3_moe' in applied_patches:
            restore_deepseek_v3_moe_patch()

        raise e

def load(
    path_or_hf_repo: str,
    tokenizer_config: Dict = {},
    dtype: torch.dtype = torch.half,
    device_map: set = {"cuda:0"},
    seqlen: int = 2048,
    model_config: Dict = {},
    requires_grad: bool = False
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
        model_config (str, optional): Model configurations for "AutoModelForCausalLM.from_pretrained".
        requires_grad (bool): Set if the model requires gradient. It can be set to True for training.

    Returns:
        Tuple[nn.Module, PreTrainedTokenizer, AutoConfig]: A tuple containing the loaded model and tokenizer.

    Raises:
        FileNotFoundError: If the config file or safetensors are not found.
        ValueError: If the model class or args class are not found.
        NotImplementedError: If the LoRA layers functionality is not implemented.
    """
    if not ENGINE_AVAILABLE:
        raise Exception("Error: Bitorch Engine layers are not available.")

    model_path = get_model_path(path_or_hf_repo)
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)

    layer_mode, bits, group_size = get_layer_mode(path_or_hf_repo, config)

    model = load_model(model_path, config, layer_mode, bits, group_size, dtype, device_map, seqlen, model_config, requires_grad)
    tokenizer = AutoTokenizer.from_pretrained(path_or_hf_repo, **tokenizer_config)

    return model, tokenizer, config


def generate(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    prompt: str,
    temp: float = 0.8,
    max_tokens: int = 100,
    verbose: bool = False,
    top_p: float = 0.95,
    gen_mode: TextGenMode = TextGenMode.SEQUENCE,
    repetition_penalty: Optional[float] = 1.5,
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
        top_p (TextGenMode): Indicates if generate text sequence or token by token generation.
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

    if verbose:
        print("=" * 50)
        print(f"Prompt: {prompt}")
        tic = time.perf_counter()

    add_special_tokens = tokenizer.bos_token is None or not prompt.startswith(
        tokenizer.bos_token
    )

    # Encode the prompt text to tensor
    input_ids = tokenizer.encode(prompt, return_tensors="pt", add_special_tokens=add_special_tokens).to(device)

    with torch.no_grad():
        if verbose:
            print(f"generating ... ")

        if gen_mode == TextGenMode.SEQUENCE:
            output_sequences = model.generate(
                input_ids=input_ids,
                temperature=temp,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                use_cache=True,
                do_sample=True,
                max_new_tokens=max_tokens,
                top_k=top_k,
                output_scores=output_scores,
                return_dict_in_generate=return_dict_in_generate,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id
            )

            full_sequence = tokenizer.decode(output_sequences['sequences'][0], clean_up_tokenization_spaces=True)

            original_prompt_text = tokenizer.decode(input_ids[0], clean_up_tokenization_spaces=True)
            generated_sequence = full_sequence[len(original_prompt_text):]

            if verbose:
                print(f"{generated_sequence}")

        elif gen_mode == TextGenMode.TOKEN:
            generated_tokens = []
            current_input_ids = input_ids

            for _ in range(max_tokens):
                output_sequences = model.generate(
                    input_ids=current_input_ids,
                    temperature=temp,
                    top_p=top_p,
                    repetition_penalty=repetition_penalty,
                    use_cache=True,
                    do_sample=True,
                    max_new_tokens=1,  # Generate one token at a time
                    top_k=top_k,
                    output_scores=output_scores,
                    return_dict_in_generate=return_dict_in_generate,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id
                )

                new_token_id = output_sequences['sequences'][0][-1]
                new_token = tokenizer.decode([new_token_id], clean_up_tokenization_spaces=True)
                generated_tokens.append(new_token)

                current_input_ids = output_sequences['sequences']

                if verbose:
                    print(new_token, end='', flush=True)

                if new_token_id == tokenizer.eos_token_id:
                    break

            generated_sequence = "".join(generated_tokens)

    if verbose:
        gen_duration = time.perf_counter() - tic
        gen_tps = max_tokens / gen_duration if gen_duration > 0 else 0
        print(f"\ngeneration: {gen_tps:.2f} token/s")

    return generated_sequence