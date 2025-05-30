import peft.peft_model
from peft.tuners import lora
from peft.utils import _get_submodules, PeftType

from green_bit_llm.sft.peft_utils.gba_lora import dispatch_gba


class GBALoraModel(lora.LoraModel):
    """
    A specialized version of LoraModel for low-rank adaptation. This class overrides the method to create new modules specifically tailored
    to GBA needs, by selecting appropriate backend functions to handle LoRA layers.
    """
    @staticmethod
    def _create_new_module(lora_config, adapter_name, target, **kwargs):
        """
        Creates a new module based on the provided configuration for LoRA and the type of target module.
        This method selects the correct dispatch function for integrating LoRA into the specified model layer.
        If no suitable module can be found, it raises an error.

        Args:
            lora_config: Configuration parameters for the LoRA adaptation.
            adapter_name: Identifier for the LoRA adapter.
            target: The target neural network layer to which LoRA should be applied.
            **kwargs: Additional keyword arguments.

        Returns:
            new_module: A new module with LoRA adaptation applied, or raises an error if the target is unsupported.

        Raises:
            ValueError: If the target module type is not supported by the currently available dispatchers.
        """
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
    """
    Replaces the existing LoRA model in the PEFT framework with the GBA-enhanced LoRA model.
    This function patches the model mapping in PEFT to use `GBALoraModel` for LoRA configurations.
    """
    peft.peft_model.PEFT_TYPE_TO_MODEL_MAPPING[PeftType.LORA] = GBALoraModel
