# Inference Package for GreenBitAI's Low-bit LLMs

## Overview

This package demonstrates the capabilities of [GreenBitAI's low-bit large language models (LLMs)](https://huggingface.co/GreenBitAI) through two main features:
1. Simple generation with `sim_gen.py` script.
2. A command-line interface (CLI) based chat demo using the `chat_cli.py` script.

Both tools are designed for efficient natural language processing, enabling quick setups and responses using local models.

## Installation

Please follow the [main installation instructions](../../README.md#installation) for how to install the packages required to run this inference package.
Further packages should not be required.

## Usage

### Simple Generation

Run the simple generation script as follows:

```bash
CUDA_VISIBLE_DEVICES=0 python -m green_bit_llm.inference.sim_gen --model GreenBitAI/Qwen-1.5-1.8B-layer-mix-bpw-3.0 --max-tokens 100 --use-flash-attention-2 --ignore-chat-template --prompt "The meaning of life is"
```

This command generates text based on the provided prompt using the specified GreenBitAI model.

### CLI-Based Chat Demo

To start the chat interface:

```bash
CUDA_VISIBLE_DEVICES=0 python -m green_bit_llm.inference.chat_cli --model GreenBitAI/Qwen-1.5-1.8B-Chat-layer-mix-bpw-2.2 --use-flash-attention-2 --multiline --mouse
```
This launches a rich command-line interface for interactive chatting.

## License
- The scripts `conversation.py`, `chat_base.py`, and `chat_cli.py` have been modified from their original versions found in [FastChat-serve](https://github.com/lm-sys/FastChat/tree/main/fastchat/serve), which are released under the [Apache 2.0 License](https://github.com/lm-sys/FastChat/tree/main/LICENSE). 
- We release our changes and additions to these files under the [Apache 2.0 License](../../LICENSE).
