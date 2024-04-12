# Inference

## Overview

This project demonstrates a quick setup for a chat interface using [GreenBitAI's low-bit models](https://huggingface.co/collections/GreenBitAI/) and [FastChat](https://github.com/lm-sys/FastChat)-CLI tool. 
It enables conversations using a local model creating an efficient and accessible environment for deploying chat applications.

## Installation


```bash
pip install pillow requests prompt_toolkit rich
```
Installation involves two main steps: setting up the gbx_lm package and installing FastChat along with its dependencies.


Upon completing these steps, the worker should be operational and accessible via a local URL: http://0.0.0.0:7860. Open this URL in your preferred web browser to begin interacting with your local MLX LLM. Enjoy your conversations!

## License
- `conversation.py`, `chat_base.py` and `chat_cli.py` released under the [Apache 2.0 License](https://github.com/lm-sys/FastChat/tree/main/LICENSE) in [FastChat-serve](https://github.com/lm-sys/FastChat/tree/main/fastchat/serve).
- We release our changes and additions to these files under the [Apache 2.0 License](../LICENSE).