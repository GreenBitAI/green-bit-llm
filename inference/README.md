# Inference Package for GreenBitAI's Low-bit LLMs

## Overview

This package demonstrates the capabilities of [GreenBitAI's low-bit large language models (LLMs)](https://huggingface.co/GreenBitAI) through two main features:
1. Simple generation with `sim_gen.py` script.
2. A command-line interface (CLI) based chat demo using the `chat_cli.py` script.

Both tools are designed for efficient natural language processing, enabling quick setups and responses using local models.

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

For the CLI-based chat demo, install the following additional libraries:

```bash
pip install pillow requests prompt_toolkit rich
```

## Usage

### Simple Generation

Run the simple generation script as follows:

```bash
CUDA_VISIBLE_DEVICES=0 python inference/sim_gen.py --model GreenBitAI/Qwen-1.5-1.8B-layer-mix-bpw-3.0 --max-tokens 100 --use-flash-attention-2 --ignore-chat-template
```

This command generates text based on the provided prompt using the specified GreenBitAI model.

### CLI-Based Chat Demo

To start the chat interface:

```bash
CUDA_VISIBLE_DEVICES=0 python inference/chat_cli.py --model-path GreenBitAI/yi-6b-chat-w4a16g256 --debug --use-flash-attention-2 --multiline --mouse
```
This launches a rich command-line interface for interactive chatting.

## License
- The scripts `conversation.py`, `chat_base.py`, and `chat_cli.py` have been modified from their original versions found in [FastChat-serve](https://github.com/lm-sys/FastChat/tree/main/fastchat/serve), which are released under the [Apache 2.0 License](https://github.com/lm-sys/FastChat/tree/main/LICENSE). 
- We release our changes and additions to these files under the [Apache 2.0 License](../LICENSE).