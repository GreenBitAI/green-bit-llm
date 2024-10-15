# Local RAG Demo

## Overview

This project demonstrates a local implementation of Retrieval-Augmented Generation (RAG) using GreenBit models, including GreenBitPipeline, ChatGreenBit, and GreenBitEmbeddings. It showcases features such as document loading, text splitting, vector store creation, and various natural language processing tasks in a CUDA environment.

## Features

- Document loading from web sources
- Text splitting for efficient processing
- Vector store creation using BERT embeddings
- Rap battle simulation
- Document summarization
- Question answering
- Question answering with retrieval

## Installation

1. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

2. Ensure you have a CUDA-compatible environment set up on your system.

## Usage

Run the main script to execute all tasks:

```
python run.py --model "GreenBitAI/Llama-3-8B-instruct-layer-mix-bpw-4.0-mlx" \
               --embedding_model "sentence-transformers/all-mpnet-base-v2" \
               --query "What are the core method components of GraphRAG?" \
               --max_tokens 300 \
               --web_source "https://www.microsoft.com/en-us/research/blog/graphrag-unlocking-llm-discovery-on-narrative-private-data/" \
               --device "cuda:0"
```

This will perform the following tasks:
1. Initialize the model and prepare data
2. Simulate a rap battle
3. Summarize documents based on a question
4. Perform question answering
5. Perform question answering with retrieval

## Note

This implementation uses GreenBit models, which are compatible with Hugging Face's transformers library and optimized for CUDA environments. Make sure you have the appropriate CUDA setup and GreenBit model files before running the demo.