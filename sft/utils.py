import torch

from colorama import init, Fore, Style
init(autoreset=True)

try:
    from bitorch_engine.layers.qlinear.nbit import MPQLinearBase
except ModuleNotFoundError as e:
    raise Exception(f"Error occurred while importing Bitorch Engine module '{str(e)}'.")


def str_to_torch_dtype(dtype: str):
    """Get torch dtype from the input data type string."""
    if dtype is None:
        return None
    elif dtype == "float":
        return torch.float
    elif dtype == "half":
        return torch.float16
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")


def create_device_map(cuda_device_id):
    ids = cuda_device_id.split(',')
    # Create strings in the format "cuda:x" for each ID and put them into the collection
    device_map = {f"cuda:{id}" for id in ids}
    return device_map


def get_learning_rate(lr_bit, galore, default_lr_galore, default_lr):
    """Adaptivly get the learning rate value from the input setting parameters."""
    if lr_bit > 0:
        return lr_bit
    return default_lr_galore if galore else default_lr


def create_param_groups(model, args, betas=(0.9, 0.999), lr_galore=1e-4, lr_adamw8b=5e-3, lr_default=1e-5):
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
    if hasattr(args, 'lora_rank'): # peft
        params_groups = []

        # Create list of peft parameters
        params_lora = [p for n, p in model.named_parameters() if "lora" in n]

        params_group_lora = {'params': params_lora, 'lr': args.lr_fp, 'betas': betas}

        params_groups.append(params_group_lora)

    elif hasattr(args, 'tune_qweight_only'): # full parameter finetune
        params_2_bit = []
        params_4_bit = []

        regular_trainable_numel = []
        qweight_trainable_numel = []
        trainable_numel = []

        for module_name, module in model.named_modules():
            if issubclass(type(module), MPQLinearBase):
                if module.w_bit == 2:
                    params_2_bit.append(module.qweight)
                    qweight_trainable_numel.append(int(module.qweight.numel() * 32 / 2))
                elif module.w_bit == 4:
                    params_4_bit.append(module.qweight)
                    qweight_trainable_numel.append(int(module.qweight.numel() * 32 / 4))
                else:
                    raise Exception(f"Error: Invalid qweight bit width: '{module.w_bit}'.")

        id_2bit_params = [id(p) for p in params_2_bit]
        id_4bit_params = [id(p) for p in params_4_bit]
        # Concatenate IDs to form a single list
        excluded_ids = id_2bit_params + id_4bit_params

        # Create list of regular parameters excluding 2-bit and 4-bit params
        params_regular = [p for p in model.parameters() if id(p) not in excluded_ids]
        for param in params_regular:
            if param.requires_grad:
                regular_trainable_numel.append(param.numel())

        lr_2 = get_learning_rate(
            args.lr_2bit,
            args.galore,
            lr_adamw8b if 'adamw8bit' in args.optimizer.lower() else lr_galore,
            lr_default)
        lr_4 = get_learning_rate(
            args.lr_4bit,
            args.galore,
            lr_adamw8b if 'adamw8bit' in args.optimizer.lower() else lr_galore,
            lr_default)

        params_group_2bit = {'params': params_2_bit, 'lr': lr_2, 'betas': betas}
        params_group_4bit = {'params': params_4_bit, 'lr': lr_4, 'betas': betas}
        params_group_regular = {'params': params_regular, 'lr': args.lr_fp, 'betas': betas}

        # Optionally add extra settings from command line arguments
        if args.galore:
            galore_settings = {
                'rank': args.galore_rank,
                'update_proj_gap': args.galore_update_proj_gap,
                'scale': args.galore_scale,
                'proj_type': args.galore_proj_type
            }
            params_group_2bit.update(galore_settings)
            params_group_4bit.update(galore_settings)

        param_groups = [
            params_group_2bit,
            params_group_4bit
        ]

        trainable_numel = qweight_trainable_numel
        if not args.tune_qweight_only:
            param_groups.append(params_group_regular)
            trainable_numel += regular_trainable_numel

        total_numel = []
        total_parameters = list(model.parameters())
        for param in total_parameters:
            if not hasattr(param, "qweight"):
                total_numel.append(param.numel())
        total_numel += qweight_trainable_numel

        print(Style.BRIGHT + Fore.CYAN +
            f"Info: trainable params: {sum(trainable_numel):,d} || "
            f"all params: {sum(total_numel):,d} || "
            f"trainable%: {100 * sum(trainable_numel) / sum(total_numel):.4f}"
        )
    else:
        raise Exception("Error: invalid use case in creating param_group.")

    return param_groups


def num_parameters(self, only_trainable: bool = False, exclude_embeddings: bool = False) -> int:
    """
    Get number of (optionally, trainable or non-embeddings) parameters in the module.

    Args:
        only_trainable (`bool`, *optional*, defaults to `False`):
            Whether or not to return only the number of trainable parameters

        exclude_embeddings (`bool`, *optional*, defaults to `False`):
            Whether or not to return only the number of non-embeddings parameters

    Returns:
        `int`: The number of parameters.
    """

    if exclude_embeddings:
        embedding_param_names = [
            f"{name}.weight" for name, module_type in self.named_modules() if isinstance(module_type, nn.Embedding)
        ]
        total_parameters = [
            parameter for name, parameter in self.named_parameters() if name not in embedding_param_names
        ]
    else:
        total_parameters = list(self.parameters())

    total_numel = []
    is_loaded_in_4bit = getattr(self, "is_loaded_in_4bit", False)

    if is_loaded_in_4bit:
        if is_bitsandbytes_available():
            import bitsandbytes as bnb
        else:
            raise ValueError(
                "bitsandbytes is not installed but it seems that the model has been loaded in 4bit precision, something went wrong"
                " make sure to install bitsandbytes with `pip install bitsandbytes`. You also need a GPU. "
            )

    for param in total_parameters:
        if param.requires_grad or not only_trainable:
            # For 4bit models, we need to multiply the number of parameters by 2 as half of the parameters are
            # used for the 4bit quantization (uint8 tensors are stored)
            if is_loaded_in_4bit and isinstance(param, bnb.nn.Params4bit):
                if hasattr(param, "element_size"):
                    num_bytes = param.element_size()
                elif hasattr(param, "quant_storage"):
                    num_bytes = param.quant_storage.itemsize
                else:
                    num_bytes = 1
                total_numel.append(param.numel() * 2 * num_bytes)
            else:
                total_numel.append(param.numel())

    print(
        f"trainable params: {trainable_params:,d} || "
        f"all params: {all_param:,d} || "
        f"trainable%: {100 * trainable_params / all_param:.4f}"
    )



