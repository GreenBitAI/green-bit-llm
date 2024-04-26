# Copyright 2023-present the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from __future__ import annotations

from typing import Any, Optional

ENGINE_AVAILABLE=True
try:
    from bitorch_engine.layers.qlinear.nbit import MPQLinearBase
    from bitorch_engine.layers.qlinear.nbit.cuda import MPQLinearCuda, MBWQLinearCuda
except ModuleNotFoundError as e:
    ENGINE_AVAILABLE = False
    raise Exception(f"Error occurred while importing Bitorch Engine module '{str(e)}'.")

import torch
import torch.nn as nn
from transformers.pytorch_utils import Conv1D

from peft.tuners.tuners_utils import BaseTunerLayer, check_adapters_to_merge
from peft.tuners.lora import LoraLayer


class GBALoraLayer(LoraLayer):
    """
    GBALoraLayer class extends LoraLayer to support Gradient-Based Adapter tuning for various model layers.
    It maintains lists of both LoRA-specific parameters and other adapter-related parameters.
    """

    # All names of layers that may contain (trainable) adapter weights
    adapter_layer_names = ("lora_A", "lora_B", "lora_embedding_A", "lora_embedding_B")
    # All names of other parameters that may contain adapter-related parameters
    other_param_names = ("r", "lora_alpha", "scaling", "lora_dropout")

    def __init__(self, base_layer: nn.Module, **kwargs) -> None:
        """
        Initializes a GBALoraLayer instance.
        Args:
            base_layer: The underlying neural network layer that LoRA is being applied to.
            **kwargs: Additional keyword arguments for customization.

        This method initializes adapter components, configures the underlying base layer, and sets the
        feature sizes based on the base layer type.
        """
        self.base_layer = base_layer
        self.r = {}
        self.lora_alpha = {}
        self.scaling = {}
        self.lora_dropout = nn.ModuleDict({})
        self.lora_A = nn.ModuleDict({})
        self.lora_B = nn.ModuleDict({})
        # For Embedding layer
        self.lora_embedding_A = nn.ParameterDict({})
        self.lora_embedding_B = nn.ParameterDict({})
        # Mark the weight as unmerged
        self._disable_adapters = False
        self.merged_adapters = []
        self.use_dora: dict[str, bool] = {}
        self.lora_magnitude_vector: Optional[torch.nn.ParameterDict] = None  # for DoRA
        self._caches: dict[str, Any] = {}
        self.kwargs = kwargs

        base_layer = self.get_base_layer()
        if isinstance(base_layer, nn.Linear):
            in_features, out_features = base_layer.in_features, base_layer.out_features
        elif isinstance(base_layer, nn.Conv2d):
            in_features, out_features = base_layer.in_channels, base_layer.out_channels
        elif isinstance(base_layer, nn.Embedding):
            in_features, out_features = base_layer.num_embeddings, base_layer.embedding_dim
        elif isinstance(base_layer, Conv1D):
            in_features, out_features = (
                base_layer.weight.ds_shape if hasattr(base_layer.weight, "ds_shape") else base_layer.weight.shape
            )
        elif hasattr(base_layer, "infeatures") and hasattr(base_layer, "outfeatures"):
            # QuantLinear
            in_features, out_features = base_layer.infeatures, base_layer.outfeatures
        elif hasattr(base_layer, "input_size") and hasattr(base_layer, "output_size"):
            # Megatron ColumnParallelLinear,RowParallelLinear
            in_features, out_features = base_layer.input_size, base_layer.output_size
        elif hasattr(base_layer, "codebooks") and base_layer.__class__.__name__ == "QuantizedLinear":
            # AQLM QuantLinear
            in_features, out_features = base_layer.in_features, base_layer.out_features
        elif hasattr(base_layer, "w_bit") and base_layer.__class__.__name__ == "WQLinear_GEMM":
            # Awq layers
            in_features, out_features = base_layer.in_features, base_layer.out_features
        elif base_layer.__class__.__name__ == "MBWQLinearCuda":
            in_features, out_features = base_layer.in_channels, base_layer.out_channels
        elif base_layer.__class__.__name__ == "MPQLinearCuda":
            in_features, out_features = base_layer.in_channels, base_layer.out_channels
        else:
            raise ValueError(f"Unsupported layer type {type(base_layer)}")

        self.in_features = in_features
        self.out_features = out_features


class GBALoraLinear(torch.nn.Module, GBALoraLayer):
    """
    Implements a LoRA (Low-Rank Adaptation) module integrated into a dense linear layer.
    This class extends functionality by allowing modifications to the layer through
    low-rank matrices to efficiently adapt large pre-trained models without extensive retraining.
    """
    # Lora implemented in a dense layer
    def __init__(
        self,
        base_layer: torch.nn.Module,
        adapter_name: str,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        init_lora_weights: bool = True,
        use_rslora: bool = False,
        use_dora: bool = False,
        **kwargs,
    ) -> None:
        """
        Initializes the LoRA adapted layer with specific parameters and configurations.

        Parameters:
            base_layer (torch.nn.Module): The original base layer to which LoRA adjustments are applied.
            adapter_name (str): The name of the adapter for identification.
            r (int): The rank of the low-rank approximation matrices.
            lora_alpha (int): Scaling factor for the LoRA parameters.
            lora_dropout (float): Dropout rate for regularization during training.
            init_lora_weights (bool): Whether to initialize LoRA weights upon creation.
            use_rslora (bool): Indicates whether to use rank-stabilized LoRA.
            use_dora (bool): Indicates whether to use dynamic orthogonal regularization adapter.
        """
        super().__init__()
        GBALoraLayer.__init__(self, base_layer)
        self.fan_in_fan_out = False

        self._active_adapter = adapter_name
        self.update_layer(
            adapter_name,
            r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            init_lora_weights=init_lora_weights,
            use_rslora=use_rslora,
            use_dora=use_dora,
        )

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """
        Defines the computation performed at every call. Applies the base layer computation
        and modifies the output using the configured LoRA parameters.

        Parameters:
            x (torch.Tensor): The input tensor to process.

        Returns:
            torch.Tensor: The output tensor after applying the LoRA adaptation.
        """
        self._check_forward_args(x, *args, **kwargs)
        adapter_names = kwargs.pop("adapter_names", None)

        result = self.base_layer(x, *args, **kwargs)
        # As per Tim Dettmers, for 4bit, we need to defensively clone here.
        # The reason is that in some cases, an error can occur that backprop
        # does not work on a manipulated view. This issue may be solved with
        # newer PyTorch versions but this would need extensive testing to be
        # sure.

        for active_adapter in self.active_adapters:
            if active_adapter not in self.lora_A.keys():
                continue
            lora_A = self.lora_A[active_adapter]
            lora_B = self.lora_B[active_adapter]
            dropout = self.lora_dropout[active_adapter]
            scaling = self.scaling[active_adapter]

            requires_conversion = not torch.is_autocast_enabled()
            if requires_conversion:
                expected_dtype = result.dtype
                x = x.to(lora_A.weight.dtype)

            if not self.use_dora[active_adapter]:
                output = lora_B(lora_A(dropout(x))) * scaling
            else:
                output = self._apply_dora(x, lora_A, lora_B, scaling, active_adapter)
            if requires_conversion:
                output = output.to(expected_dtype)

            result = result + output

        return result

    def __repr__(self) -> str:
        """
        Provides a string representation of the module, enhancing the default
        representation with a prefix to identify it as a LoRA-adapted layer.
        """
        rep = super().__repr__()
        return "lora." + rep


def dispatch_gba(target: torch.nn.Module, adapter_name: str, **kwargs):
    new_module = None

    if isinstance(target, BaseTunerLayer):
        target_base_layer = target.get_base_layer()
    else:
        target_base_layer = target

    if ENGINE_AVAILABLE and issubclass(type(target_base_layer), MPQLinearBase):
        new_module = GBALoraLinear(target_base_layer, adapter_name, **kwargs)

    return new_module
