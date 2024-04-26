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

    This function also prints out the number of trainable params and all params.

    Args:
        model (nn.Module): The neural network model.
        args (argparse.ArgumentParser): Command line arguments for additional parameters.

    Returns:
        List[dict]: A list of dictionaries where each dictionary contains a parameter group.
    """
    params_2_bit = []
    params_4_bit = []

    regular_trainable_numel = []
    qweight_trainable_numel = []
    total_numel = []
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

    total_parameters = list(model.parameters())
    for param in total_parameters:
        if not hasattr(param, "qweight"):
            total_numel.append(param.numel())
    total_numel += qweight_trainable_numel

    if hasattr(args, 'lora_rank'): # peft
        param_groups = []

        # Create list of peft parameters
        params_lora = [p for n, p in model.named_parameters() if "lora" in n]

        for param in params_lora:
            if param.requires_grad:
                trainable_numel.append(param.numel())

        params_group_lora = {'params': params_lora, 'lr': args.lr_fp, 'betas': betas}

        param_groups.append(params_group_lora)

    elif hasattr(args, 'tune_qweight_only'): # full parameter finetune

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
            1e-3 if 'adamw8bit' in args.optimizer.lower() else lr_default
        )

        lr_4 = get_learning_rate(
            args.lr_4bit,
            args.galore,
            lr_adamw8b if 'adamw8bit' in args.optimizer.lower() else lr_galore,
            1e-3 if 'adamw8bit' in args.optimizer.lower() else lr_default
        )

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
    else:
        raise Exception("Error: invalid use case in creating param_group.")

    # print out trainable params info
    print(Style.BRIGHT + Fore.CYAN +
        f"Info: trainable params: {sum(trainable_numel):,d} || "
        f"all params: {sum(total_numel):,d} || "
        f"trainable%: {100 * sum(trainable_numel) / sum(total_numel):.4f}"
    )

    return param_groups



