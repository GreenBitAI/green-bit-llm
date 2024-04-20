# Finetuning GreenBitAI's Low-bit LLMs

## Overview

This package demonstrates the capabilities of [GreenBitAI's low-bit large language models (LLMs)](https://huggingface.co/GreenBitAI) through two main features:
1. Full-parameter finetuning using quantized LLMs.
2. Parameter efficient finetuning


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

## Usage

### Full-parameter finetuning

Run the script as follows to fine-tune the quantized weights of the model on the target dataset. 
The '--tune-qweight-only' parameter determines whether to fine-tune only the quantized weights or all weights, including non-quantized ones.

```bash
CUDA_VISIBLE_DEVICES=0 python -m sft.finetune --model GreenBitAI/Qwen-1.5-1.8B-layer-mix-bpw-3.0 --dataset tatsu-lab/alpaca --galore --tune-qweight-only
```
For more information about this parameter, please refer to:
```bash
python -m sft.finetune --help
```

### Parameter efficient finetuning

```bash
CUDA_VISIBLE_DEVICES=0 python ...
```
...

## License
We release our changes and additions to these files under the [Apache 2.0 License](../LICENSE).