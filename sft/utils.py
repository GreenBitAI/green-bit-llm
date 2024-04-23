import torch


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