# GreenBit Langchain Demos

## Overview

GreenBit Langchain Demos showcase the integration of GreenBit language models with the Langchain framework, enabling powerful and flexible natural language processing capabilities.

## Installation

### Step 1: Install the gbx_lm Package

```bash
pip install green-bit-llm
```

### Step 2: Install Langchain Package

```bash
pip install langchain-core
```

If you want to use RAG demo, please make sure that the `sentence_transformers` python package has been installed. 

```bash
pip install sentence-transformers
```

Ensure your system has Python3 and pip installed before proceeding.

## Usage

### Basic Example

Here's a basic example of how to use the GreenBit Langchain integration:

```python
from langchain_core.messages import HumanMessage
from green_bit_llm.langchain import GreenBitPipeline, ChatGreenBit
import torch

pipeline = GreenBitPipeline.from_model_id(
    model_id="GreenBitAI/Llama-3-8B-instruct-layer-mix-bpw-4.0",
    device="cuda:0",
    model_kwargs={"dtype": torch.half, "device_map": 'auto', "seqlen": 2048, "requires_grad": False},
    pipeline_kwargs={"max_new_tokens": 100, "temperature": 0.7},
)

chat = ChatGreenBit(llm=pipeline)

# normal generation
response = chat.invoke("What is the capital of France?")
print(response)

# stream generation
for chunk in chat.stream([HumanMessage(content="Tell me a story about a brave knight.")]):
    print(chunk.message.content, end="", flush=True)

```
