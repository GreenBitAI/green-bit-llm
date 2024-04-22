import torch
from torch import nn

from peft.tuners import lora
from peft.tuners.lora import LoraLayer
from peft.utils import _get_submodules, PeftType

from .gba_lora import dispatch_gba

class GBALoraModel(lora.LoraModel):
    @staticmethod
    def _create_new_module(lora_config, adapter_name, target, **kwargs):
        # Collect dispatcher functions to decide what backend to use for the replaced LoRA layer. The order matters,
        # because the first match is always used. Therefore, the default layers should be checked last.
        dispatchers = []

        dispatchers.append(dispatch_gba)

        dispatchers.extend(
            [dispatch_gba]
        )

        new_module = None
        for dispatcher in dispatchers:
            new_module = dispatcher(target, adapter_name, lora_config=lora_config, **kwargs)
            if new_module is not None:  # first match wins
                break

        if new_module is None:
            # no module could be matched
            raise ValueError(
                f"Target module {target} is not supported. Currently, only the following modules are supported: "
                "`torch.nn.Linear`, `torch.nn.Embedding`, `torch.nn.Conv2d`, `transformers.pytorch_utils.Conv1D`."
            )

        return new_module

def replace_peft_lora_model_with_gba_lora_model():
    import peft.peft_model
    peft.peft_model.PEFT_TYPE_TO_MODEL_MAPPING[PeftType.LORA] = GBALoraModel
