# Finetuning GreenBitAI's Low-bit LLMs

## Overview

This package demonstrates the capabilities of [GreenBitAI's low-bit large language models (LLMs)](https://huggingface.co/GreenBitAI) through two main features:
1. Full-parameter fine-tuning using quantized LLMs.
2. Parameter efficient fine-tuning


## Installation

Please follow the [main installation instructions](../../README.md#installation) for how to install the packages required to run this inference package.
Afterward, install the following additional libraries:

```bash
pip install trl
pip install -U git+https://github.com/huggingface/peft.git
```

If you want to use a **8-bit customized optimizer** with the gradient low-rank projection for maximizing memory savings, you will also need to install the following package:

```bash
pip install bitsandbytes galore-torch
```

## Usage

### LLMs

We have released over 200 highly precise 2.2/2.5/3/4-bit models across the modern LLM family, featuring LLaMA 2/3, 01-Yi, Qwen, Mistral, Phi-3, and more. Currently, only layer-mix quantized models are supported for sft. In addition to our low-bit models, green-bit-llm is fully compatible with the AutoGPTQ series of 4-bit quantization and compression models.

Happy scaling low-bit LLMs with more data!

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


### Full-parameter fine-tuning

Run the script as follows to fine-tune the quantized weights of the model on the target dataset. 
The **--tune-qweight-only** parameter determines whether to fine-tune only the quantized weights or all weights, including non-quantized ones.

```bash
CUDA_VISIBLE_DEVICES=0 python -m green_bit_llm.sft.finetune --model GreenBitAI/Qwen-1.5-1.8B-layer-mix-bpw-3.0 --dataset tatsu-lab/alpaca --tune-qweight-only

# AutoGPTQ model Q-SFT
CUDA_VISIBLE_DEVICES=0 python -m green_bit_llm.sft.finetune --model astronomer/Llama-3-8B-Instruct-GPTQ-4-Bit --dataset tatsu-lab/alpaca --tune-qweight-only --batch-size 1
```
If you want to further save memory, we also support [Galore](https://github.com/jiaweizzhao/GaLore): a memory-efficient low-rank training strategy. 
You just need to add the **--galore** parameter in your command line. However, it's important to note that Galore requires the computation of projection matrices for the gradients, which will incur additional time costs. 
You can think of this as a trade-off strategy where time is exchanged for space.
To select the "AdamW8bit" optimizer, simply add "--optimizer AdamW8bit" to your command line.

### Parameter efficient fine-tuning

```bash
CUDA_VISIBLE_DEVICES=0 python -m green_bit_llm.sft.peft_lora --model GreenBitAI/Qwen-1.5-1.8B-layer-mix-bpw-3.0 --dataset tatsu-lab/alpaca --lr-fp 1e-6

CUDA_VISIBLE_DEVICES=0 python -m green_bit_llm.sft.peft_lora --model astronomer/Llama-3-8B-Instruct-GPTQ-4-Bit --dataset tatsu-lab/alpaca --lr-fp 1e-6
```

### 0-Shot Evaluation of Q-SFT 

The 0-shot evaluations of quantized Llama 3 8B model under different fine-tuning settings are listed as an example. **Q-SFT** indicates quantized surpervised-finetuning.

| Task          |   Bpw    |  Llama 3 8B Base  |  Llama 3 8B + LoRA  | Llama 3 8B Q-SFT + Galore  | Llama 3 8B + Q-SFT |
|:-------------:|:--------:|:-----------------:|:-------------------:|:--------------------------:|:------------------:|
|     PIQA      |   2.2    |       0.72        |        0.75         |            0.75            |        0.75        |
|               |   2.5    |       0.74        |        0.77         |            0.76            |        0.76        |
|               |   3.0    |       0.76        |        0.78         |            0.78            |        0.79        |
|     BoolQ     |   2.2    |       0.74        |        0.77         |            0.77            |        0.78        |
|               |   2.5    |       0.75        |        0.76         |            0.76            |        0.78        |
|               |   3.0    |       0.78        |        0.80         |            0.79            |        0.80        |
|     Winogr.   |   2.2    |       0.67        |        0.68         |            0.68            |        0.67        |
|               |   2.5    |       0.68        |        0.69         |            0.69            |        0.69        |
|               |   3.0    |       0.70        |        0.71         |            0.71            |        0.71        |
|     ARC-E     |   2.2    |       0.73        |        0.77         |            0.76            |        0.75        |
|               |   2.5    |       0.76        |        0.77         |            0.77            |        0.76        |
|               |   3.0    |       0.77        |        0.79         |            0.79            |        0.79        |
|     ARC-C     |   2.2    |       0.39        |        0.46         |            0.45            |        0.45        |
|               |   2.5    |       0.41        |        0.44         |            0.43            |        0.43        |
|               |   3.0    |       0.44        |        0.49         |            0.47            |        0.49        |
|     WiC       |   2.2    |       0.50        |        0.50         |            0.50            |        0.50        |
|               |   2.5    |       0.51        |        0.50         |            0.52            |        0.51        |
|               |   3.0    |       0.52        |        0.52         |            0.57            |        0.60        |
|     Avg       |   2.2    |       0.62        |        0.65         |            0.65            |        0.65        |
|               |   2.5    |       0.64        |        0.65         |            0.65            |        0.65        |
|               |   3.0    |       0.66        |        0.68         |            0.68            |        0.69        |

Compared to traditional LoRA based fine-tuning, our approach streamlines engineering supply chain from fine-tuning to hardware deployment, while also enhancing performance.

### Current Limitations

1. Gradient clipping is currently unavailable for Full-parameter fine-tuning due to PyTorch's restrictions on the dtype of gradient tensors. The integer tensor type we use for qweight is not supported. We plan to address this issue gradually in the future.
2. Due to the need for deep modifications to the Python code related to the Int gradient tensor type, to ensure stability and safety, we currently do not support distributed or data parallel training. We plan to support this in the future. Stay tuned.


## License
- The script 'optim/adamw8bit.py' has been modified from [GaLore repository](https://github.com/jiaweizzhao/GaLore/blob/master/galore_torch/adamw8bit.py), which is released under the Apache 2.0 License.
- The script 'optim/bnb_optimizer.py' has been modified from [bitsandbytes repository](https://github.com/TimDettmers/bitsandbytes/blob/main/bitsandbytes/optim/optimizer.py), which is released under the MIT License.
- We release our changes and additions to these files under the [Apache 2.0 License](../../LICENSE).
