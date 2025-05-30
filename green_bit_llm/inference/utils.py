from transformers.generation.logits_process import (
    LogitsProcessorList,
    RepetitionPenaltyLogitsProcessor,
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
)

from .conversation import Conversation, get_conv_template

# value is the search term in model name,
# key is the name of conversation template
CONV_TEMP_DICT = {
    "llama-2": "llama-2",
    "qwen-chat": "qwen",
    "yi-chat": "yi-",
    "mistral": "mistral",
    "gemma": "gemma",
    "llama-3": "llama-3",
    "phi-3": "phi-3",
}

# Models don't use the same configuration key for determining the maximum
# sequence length.  Store them here so we can sanely check them.
# NOTE: The ordering here is important.  Some models have two of these and we
# have a preference for which value gets used.
SEQUENCE_LENGTH_KEYS = [
    "max_position_embeddings",
    "max_sequence_length",
    "seq_length",
    "max_seq_len",
    "model_max_length",
]


def is_partial_stop(output: str, stop_str: str):
    """Check whether the output contains a partial stop str."""
    for i in range(0, min(len(output), len(stop_str))):
        if stop_str.startswith(output[-i:]):
            return True
    return False

def is_sentence_complete(output: str):
    """Check whether the output is a complete sentence."""
    end_symbols = (".", "?", "!", "...", "。", "？", "！", "…", '"', "'", "”")
    return output.endswith(end_symbols)

def get_context_length(config):
    """Get the context length of a model from a huggingface model config."""
    rope_scaling = getattr(config, "rope_scaling", None)
    if rope_scaling:
        try:
            rope_scaling_factor = config.rope_scaling["factor"]
        except KeyError:
            rope_scaling_factor = 1
    else:
        rope_scaling_factor = 1

    for key in SEQUENCE_LENGTH_KEYS:
        val = getattr(config, key, None)
        if val is not None:
            return int(rope_scaling_factor * val)
    return 2048

def get_conversation_template(model_path: str) -> Conversation:
    """Get and return a specific conversation template via checking its model path/model name."""
    for key, value in CONV_TEMP_DICT.items():
        # check if model path contains the value
        if value in model_path.lower():
            return get_conv_template(key)
    raise Exception("Invalid model path: The provided model is not supported yet.")

def prepare_logits_processor(
    temperature: float, repetition_penalty: float, top_p: float, top_k: int
) -> LogitsProcessorList:
    """
    Creates and initializes a list of logits processors based on the specified parameters.
    Each processor applies a different modification to the logits during text generation,
    such as adjusting the sampling temperature, applying repetition penalties,
    or enforcing top-p and top-k constraints.

    Args:
        temperature (float): Scaling factor for logits; a value of 1.0 means no scaling.
        repetition_penalty (float): Penalty for repeated tokens to discourage repetition.
        top_p (float): The cumulative probability threshold for nucleus sampling, filters out the smallest probabilities.
        top_k (int): The number of highest probability logits to keep for top-k sampling.

    Returns:
        LogitsProcessorList: A configured list of logits processors.
    """
    processor_list = LogitsProcessorList()
    # TemperatureLogitsWarper doesn't accept 0.0, 1.0 makes it a no-op so we skip two cases.
    if temperature >= 1e-5 and temperature != 1.0:
        processor_list.append(TemperatureLogitsWarper(temperature))
    if repetition_penalty > 1.0:
        processor_list.append(RepetitionPenaltyLogitsProcessor(repetition_penalty))
    if 1e-8 <= top_p < 1.0:
        processor_list.append(TopPLogitsWarper(top_p))
    if top_k > 0:
        processor_list.append(TopKLogitsWarper(top_k))
    return processor_list

def str_to_torch_dtype(dtype: str):
    """Get torch dtype via parsing the dtype string."""
    import torch

    if dtype is None:
        return None
    elif dtype == "float32":
        return torch.float32
    elif dtype == "float16":
        return torch.half
    elif dtype == "bfloat16":
        return torch.bfloat16
    else:
        raise ValueError(f"Unrecognized dtype: {dtype}")

def is_chat_model(path):
    """Distinguish if the input model name contains keywords like '-chat-' or '-instrct-'"""
    substrings = ["-chat-", "-instruct-"]
    return any(substring in path for substring in substrings)