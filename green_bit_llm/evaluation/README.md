## Evaluation

### Installing Dependencies

Ensure your system has Python3 and pip installed before proceeding and install the following additional libraries:

```bash
pip install lm_eval==0.3.0 termcolor
```

### Command Details 

Example evaluation scripts for the GreenBitAI family of low-bit models.
- `evaluate.py` contains code to evaluate model ppl performance on `wikitext2,c4_new,ptb` tasks, and few shot ability using `lm_eval==0.3.0` library.
    - `--seed` The random seed for data loader.
    - `--model` The path to the local model directory or Hugging Face repo.
    - `--cuda-device-id` CUDA device IDs.
    - `--trust-remote-code` Enable trusting remote code for tokenize.
    - `--use-flash-attention-2` Enable using flash attention v2.
    - `--eos-token` End of sequence token for tokenizer.
    - `--seqlen` Sequence length.
    - `--eval-ppl` Evaluate LLM prediction perplexity.
    - `--ppl-tasks` Specify ppl evaluation task.
    - `--eval-few-shot` Evaluate LLM few-shot learning ability.
    - `--num-fewshot` Specify num of few shot examples for evaluation.
    - `--few-shot-tasks` Few-shot learning ability evaluation tasks.
    - `--batch-size` Batch size for few-shot evaluation.
    - `--save-dir` Specify save dir for eval results.

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
| Mistral-instruct |   `3.0/2.5/2.2`    |              `7B`              | [`GreenBitAI Mistral`](https://huggingface.co/collections/GreenBitAI/greenbitai-mistral-661f896c45da9d8b28a193a8) |                                                         |  Phi-3-instruct  |   `3.0/2.5/2.2`    |             `mini`             |   [`GreenBitAI Phi-3`](https://huggingface.co/collections/GreenBitAI/greenbitai-phi-3-6628d008cdf168398a296c92)   |
|  Qwen-1.5-Chat   | `4.0/3.0/2.5/2.2`  | `0.5B/1.8B/4B/7B/14B/32B/110B` | [`GreenBitAI Qwen 1.5`](https://huggingface.co/collections/GreenBitAI/greenbitai-qwen15-661f86ea69433f3d3062c920) |
|    01-Yi-Chat    |   `3.0/2.5/2.2`    |            `6B/34B`            |   [`GreenBitAI 01-Yi`](https://huggingface.co/collections/GreenBitAI/greenbitai-01-yi-661f88af0648daa766d5102f)   |           

### PPL Evaluation
```bash
CUDA_VISIBLE_DEVICES=0 python -m green_bit_llm.evaluation.evaluate --model GreenBitAI/Qwen-1.5-4B-layer-mix-bpw-3.0 --trust-remote-code --eval-ppl --ppl-tasks wikitext2,c4_new,ptb
```

### Few-Shot Evaluation
```bash
CUDA_VISIBLE_DEVICES=0 python -m green_bit_llm.evaluation.evaluate --model GreenBitAI/Qwen-1.5-4B-layer-mix-bpw-3.0 --trust-remote-code --batch-size 8 --few-shot-tasks wic,boolq --eval-few-shot
```

The zero-shot evaluation results for low-bit Qwen 1.5 family are listed as an example:

| HuggingFace Repository                            | Avg Accuracy | OpenBookQA | ARC Easy | Winogrande | HellaSWAG | ARC Challenge | PIQA  | BoolQ | RACE  | ANLI R1 | ANLI R2 | ANLI R3 | WiC   |
|---------------------------------------------------|--------------|------------|----------|------------|-----------|---------------|-------|-------|-------|---------|---------|---------|-------|
| `GreenBitAI/Qwen-1.5-1.8B-layer-mix-bpw-2.2`      | 0.415        | 0.218      | 0.539    | 0.586      | 0.392     | 0.260         | 0.678 | 0.622 | 0.333 | 0.333   | 0.333   | 0.336   | 0.464 |
| `GreenBitAI/Qwen-1.5-1.8B-layer-mix-bpw-2.5`      | 0.423        | 0.222      | 0.592    | 0.585      | 0.406     | 0.267         | 0.695 | 0.629 | 0.336 | 0.314   | 0.339   | 0.361   | 0.507 |
| `GreenBitAI/Qwen-1.5-1.8B-layer-mix-bpw-3.0`      | 0.438        | 0.246      | 0.576    | 0.563      | 0.413     | 0.277         | 0.694 | 0.645 | 0.352 | 0.323   | 0.336   | 0.343   | 0.492 |
| `GreenBitAI/Qwen-1.5-4B-layer-mix-bpw-2.2`        | 0.480        | 0.254      | 0.663    | 0.623      | 0.463     | 0.339         | 0.712 | 0.718 | 0.349 | 0.326   | 0.355   | 0.384   | 0.513 |
| `GreenBitAI/Qwen-1.5-4B-layer-mix-bpw-2.5`        | 0.490        | 0.266      | 0.677    | 0.629      | 0.473     | 0.365         | 0.732 | 0.717 | 0.351 | 0.372   | 0.352   | 0.360   | 0.502 |
| `GreenBitAI/Qwen-1.5-4B-layer-mix-bpw-3.0`        | 0.502        | 0.268      | 0.678    | 0.642      | 0.494     | 0.358         | 0.755 | 0.757 | 0.380 | 0.395   | 0.395   | 0.392   | 0.519 |
| `GreenBitAI/Qwen-1.5-7B-layer-mix-bpw-2.2`        | 0.513        | 0.278      | 0.669    | 0.654      | 0.504     | 0.389         | 0.741 | 0.759 | 0.376 | 0.383   | 0.410   | 0.403   | 0.517 |
| `GreenBitAI/Qwen-1.5-7B-layer-mix-bpw-2.5`        | 0.520        | 0.294      | 0.705    | 0.650      | 0.520     | 0.387         | 0.750 | 0.769 | 0.371 | 0.445   | 0.424   | 0.398   | 0.564 |
| `GreenBitAI/Qwen-1.5-7B-layer-mix-bpw-3.0`        | 0.531        | 0.292      | 0.713    | 0.654      | 0.545     | 0.405         | 0.764 | 0.807 | 0.383 | 0.424   | 0.393   | 0.414   | 0.627 |
| `GreenBitAI/Qwen-1.5-14B-layer-mix-bpw-2.5`       | 0.553        | 0.318      | 0.727    | 0.682      | 0.564     | 0.413         | 0.775 | 0.792 | 0.390 | 0.472   | 0.434   | 0.446   | 0.623 |
| `GreenBitAI/Qwen-1.5-14B-layer-mix-bpw-2.5`       | 0.568        | 0.302      | 0.734    | 0.693      | 0.583     | 0.426         | 0.785 | 0.831 | 0.395 | 0.484   | 0.443   | 0.455   | 0.686 |

