# Finetuning GreenBitAI's Low-bit LLMs

## Overview

This package demonstrates the capabilities of [GreenBitAI's low-bit large language models (LLMs)](https://huggingface.co/GreenBitAI) through two main features:
1. Full-parameter fine-tuning using quantized LLMs.
2. Parameter efficient fine-tuning


## Installation

### Prerequisites
Ensure you have Python installed on your machine. It's recommended to use a virtual environment for Python projects to manage dependencies efficiently.

### Installing Dependencies

Install green-bit-llm package using pip:

```bash
pip install green-bit-llm
```

or from source:

```bash
git clone https://github.com/GreenBitAI/green-bit-llm.git
cd green-bit-llm
pip install -r requirements.txt
```
Ensure your system has Python3 and pip installed before proceeding.

Install the following additional libraries:

```bash
pip install trl
pip install -U git+https://github.com/huggingface/peft.git
```

If you want to use a **8-bit customized optimizer** with the gradient low-rank projection for maximizing memory savings, you will also need to install the following package:

```bash
pip install bitsandbytes, galore-torch
```

## Usage

### Full-parameter fine-tuning

Run the script as follows to fine-tune the quantized weights of the model on the target dataset. 
The '--tune-qweight-only' parameter determines whether to fine-tune only the quantized weights or all weights, including non-quantized ones.

```bash
CUDA_VISIBLE_DEVICES=0 python -m sft.finetune --model GreenBitAI/Qwen-1.5-1.8B-layer-mix-bpw-3.0 --dataset tatsu-lab/alpaca --optimizer DiodeMix --tune-qweight-only --galore
```

### Parameter efficient fine-tuning

```bash
CUDA_VISIBLE_DEVICES=0 python -m sft.peft_lora --model GreenBitAI/Qwen-1.5-1.8B-layer-mix-bpw-3.0 --dataset tatsu-lab/alpaca --lr-fp 1e-6
```


### Current Limitations

1. **Gradient clipping** is currently **unavailable** for Full-parameter fine-tuning due to PyTorch's restrictions on the dtype of gradient tensors. The integer tensor type we use for qweight is not supported. We plan to address this issue gradually in the future.
2. **placeholder b**


## License
- The script 'optim/adamw8bit.py' has been modified from [GaLore repository](https://github.com/jiaweizzhao/GaLore/blob/master/galore_torch/adamw8bit.py), which is released under the Apache 2.0 License.
- The script 'optim/bnb_optimizer.py' has been modified from [bitsandbytes repository](https://github.com/TimDettmers/bitsandbytes/blob/main/bitsandbytes/optim/optimizer.py), which is released under the MIT License.
- We release our changes and additions to these files under the [Apache 2.0 License](../LICENSE).
