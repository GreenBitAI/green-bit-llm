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
from green_bit_llm.langchain import GreenBitPipeline
gb = GreenBitPipeline.from_model_id(
    model_id="GreenBitAI/Llama-3-8B-instruct-layer-mix-bpw-4.0",
    task="text-generation",
    pipeline_kwargs={"max_tokens": 100, "temp": 0.7},
    model_kwargs={"dtype": torch.half, "device_map": 'auto', "seqlen": 2048, "requires_grad": False}
)
```
