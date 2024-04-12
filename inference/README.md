# Inference Package for GreenBitAI's Low-bit LLMs

## Overview

This package demonstrates the capabilities of [GreenBitAI's low-bit large language models (LLMs)](https://huggingface.co/GreenBitAI) through two main features:
1. Simple generation with `generation.py` script.
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

Or from source:

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
python inference/generation.py --model GreenBitAI/Mistral-7B-v0.1-channel-mix-bpw-2.2 --gpus '0' --trust-remote-code --max-tokens 100 --use-flash-attention-2 --prompt 'The meaning of life is' --ignore-chat-template
```

This command generates text based on the provided prompt using the specified GreenBitAI model.

### CLI-Based Chat Demo

To start the chat interface:

```bash
python inference/chat_cli.py --model-path GreenBitAI/Mistral-Instruct-7B-v0.2-layer-mix-bpw-2.2 --style rich --debug
```

This launches a rich command-line interface for interactive chatting.



## License
- The scripts `conversation.py`, `chat_base.py`, and `chat_cli.py` have been modified from their original versions found in [FastChat-serve](https://github.com/lm-sys/FastChat/tree/main/fastchat/serve), which are released under the [Apache 2.0 License](https://github.com/lm-sys/FastChat/tree/main/LICENSE). 
- We release our changes and additions to these files under the [Apache 2.0 License](../LICENSE).