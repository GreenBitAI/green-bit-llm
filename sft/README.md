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

Run the script as follows:

```bash
CUDA_VISIBLE_DEVICES=0 python ...
```

This command ... using the specified GreenBitAI model.

### Parameter efficient finetuning

```bash
CUDA_VISIBLE_DEVICES=0 python ...
```
...

## License
We release our changes and additions to these files under the [Apache 2.0 License](../LICENSE).