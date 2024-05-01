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

### LLMs

We have released over 200 highly precise 2.2/2.5/3/4-bit models across the modern LLM family, featuring LLaMA 2/3, 01-Yi, Qwen, Mistral, Phi-3, and more.

|       Family     |        Bpw         |              Size              |                                                 HF collection id                                                  |
|:----------------:|:------------------:|:------------------------------:|:-----------------------------------------------------------------------------------------------------------------:|
|     Llama-3      |  `4.0/3.0/2.5/2.2` |            `8B/70B`            | [`GreenBitAI Llama-3`](https://huggingface.co/collections/GreenBitAI/greenbitai-llama-3-6627bc1ec6538e3922c5d81c) |
|     Llama-2      |   `3.0/2.5/2.2`    |          `7B/13B/70B`          | [`GreenBitAI Llama-2`](https://huggingface.co/collections/GreenBitAI/greenbitai-llama-2-661f87e3b073ff8e48a12834) |
|     Qwen-1.5     | `4.0/3.0/2.5/2.2`  | `0.5B/1.8B/4B/7B/14B/32B/110B` | [`GreenBitAI Qwen 1.5`](https://huggingface.co/collections/GreenBitAI/greenbitai-qwen15-661f86ea69433f3d3062c920) |
|      Phi-3       |   `3.0/2.5/2.2`    |             `mini`             |   [`GreenBitAI Phi-3`](https://huggingface.co/collections/GreenBitAI/greenbitai-phi-3-6628d008cdf168398a296c92)   |
|     Mistral      |   `3.0/2.5/2.2`    |              `7B`              | [`GreenBitAI Mistral`](https://huggingface.co/collections/GreenBitAI/greenbitai-mistral-661f896c45da9d8b28a193a8) |
|      01-Yi       |   `3.0/2.5/2.2`    |            `6B/34B`            |   [`GreenBitAI 01-Yi`](https://huggingface.co/collections/GreenBitAI/greenbitai-01-yi-661f88af0648daa766d5102f)   |
| Llama-3-instruct | `4.0/3.0/2.5/2.2`  |            `8B/70B`            | [`GreenBitAI Llama-3`](https://huggingface.co/collections/GreenBitAI/greenbitai-llama-3-6627bc1ec6538e3922c5d81c) |
| Mistral-instruct |   `3.0/2.5/2.2`    |              `7B`              | [`GreenBitAI Mistral`](https://huggingface.co/collections/GreenBitAI/greenbitai-mistral-661f896c45da9d8b28a193a8) |
|  Phi-3-instruct  |   `3.0/2.5/2.2`    |             `mini`             |   [`GreenBitAI Phi-3`](https://huggingface.co/collections/GreenBitAI/greenbitai-phi-3-6628d008cdf168398a296c92)   |
|  Qwen-1.5-Chat   | `4.0/3.0/2.5/2.2`  | `0.5B/1.8B/4B/7B/14B/32B/110B` | [`GreenBitAI Qwen 1.5`](https://huggingface.co/collections/GreenBitAI/greenbitai-qwen15-661f86ea69433f3d3062c920) |
|    01-Yi-Chat    |   `3.0/2.5/2.2`    |            `6B/34B`            |   [`GreenBitAI 01-Yi`](https://huggingface.co/collections/GreenBitAI/greenbitai-01-yi-661f88af0648daa766d5102f)   |    


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
