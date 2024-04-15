# Green-Bit-LLM

A toolkit for fine-tuning, inference, and evaluating GreenBitAI's LLMs.

## Introduction
 
This Python package uses the [Bitorch Engine](https://github.com/GreenBitAI/bitorch-engine) for efficient operations on [GreenBitAI's Low-bit Language Models (LLMs)](https://huggingface.co/GreenBitAI). 
It enables **high-performance inference** on both cloud-based and consumer-level GPUs, and supports **full-parameter fine-tuning** directly **using quantized LLMs**. 
Additionally, you can use our provided **evaluation tools** to validate the model's performance on mainstream benchmark datasets.

## Installation

### Using Pip

```bash
pip install green-bit-llm
```

### From source

simply clone the repository and install the required dependencies (for Python >= 3.9):
```bash
git clone https://github.com/GreenBitAI/green-bit-llm.git
pip install -r requirements.txt
```

### Conda

Alternatively you can also use the prepared conda environment configuration:
```bash
conda env create -f environment.yml
conda activate gbai_cuda_lm
```

## Usage
### Inference

Please see the description of the [Inference package](inference/README.md) for details.

### Evaluation

Please see the description of the [Evaluation package](evaluation/README.md) for details.

### sft

Please see the description of the [sft package](sft/README.md) for details.

## Requirements

- Python 3.x
- [Bitorch Engine](https://github.com/GreenBitAI/bitorch-engine)
- See `requirements.txt` or `environment.yml` for a complete list of dependencies

## Examples

### Simple Generation

Run the simple generation script as follows:

```bash
CUDA_VISIBLE_DEVICES=0 python -m inference.sim_gen --model GreenBitAI/Qwen-1.5-1.8B-layer-mix-bpw-3.0 --max-tokens 100 --use-flash-attention-2 --ignore-chat-template
```

### PPL Evaluation
```bash
CUDA_VISIBLE_DEVICES=0 python -m evaluation.evaluate --model GreenBitAI/Qwen-1.5-4B-layer-mix-bpw-3.0 --trust-remote-code --eval-ppl --ppl-tasks wikitext2,c4_new,ptb
```

### Run sft

```bash
```

## License
We release our codes under the [Apache 2.0 License](LICENSE).
Additionally, three packages are also partly based on third-party open-source codes. For detailed information, please refer to the description pages of the sub-projects.

