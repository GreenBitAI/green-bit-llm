# Evaluation Package for GreenBitAI's Low-bit LLMs

## Installation

Please follow the [main installation instructions](../../README.md#installation) for how to install the packages required to run this inference package.
Further packages should not be required.

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

We have released over 200 highly precise 2.2/2.5/3/4-bit models across the modern LLM family, featuring LLaMA 2/3, 01-Yi, Qwen, Mistral, Phi-3, and more. All models are quantized through a mix of precision (`Int4` and `Int2`, for easier hardware layout and deployment). 

|      Family      |        Bpw         |              Size              |                                                 HF collection_id                                                  |
|:----------------:|:------------------:|:------------------------------:|:-----------------------------------------------------------------------------------------------------------------:|
|     Llama-3      |  `4.0/3.0/2.5/2.2` |            `8B/70B`            | [`GreenBitAI Llama-3`](https://huggingface.co/collections/GreenBitAI/greenbitai-llama-3-6627bc1ec6538e3922c5d81c) |
|     Llama-2      |   `3.0/2.5/2.2`    |          `7B/13B/70B`          | [`GreenBitAI Llama-2`](https://huggingface.co/collections/GreenBitAI/greenbitai-llama-2-661f87e3b073ff8e48a12834) |
|     Qwen-1.5     | `4.0/3.0/2.5/2.2`  | `0.5B/1.8B/4B/7B/14B/32B/110B` | [`GreenBitAI Qwen 1.5`](https://huggingface.co/collections/GreenBitAI/greenbitai-qwen15-661f86ea69433f3d3062c920) |
|      Phi-3       |   `3.0/2.5/2.2`    |             `mini`             |   [`GreenBitAI Phi-3`](https://huggingface.co/collections/GreenBitAI/greenbitai-phi-3-6628d008cdf168398a296c92)   |
|     Mistral      |   `3.0/2.5/2.2`    |              `7B`              | [`GreenBitAI Mistral`](https://huggingface.co/collections/GreenBitAI/greenbitai-mistral-661f896c45da9d8b28a193a8) |
|      01-Yi       |   `3.0/2.5/2.2`    |            `6B/34B`            |   [`GreenBitAI 01-Yi`](https://huggingface.co/collections/GreenBitAI/greenbitai-01-yi-661f88af0648daa766d5102f)   |
| Llama-3-instruct | `4.0/3.0/2.5/2.2`  |            `8B/70B`            | [`GreenBitAI Llama-3`](https://huggingface.co/collections/GreenBitAI/greenbitai-llama-3-6627bc1ec6538e3922c5d81c) |
| Mistral-instruct |   `3.0/2.5/2.2`    |              `7B`              | [`GreenBitAI Mistral`](https://huggingface.co/collections/GreenBitAI/greenbitai-mistral-661f896c45da9d8b28a193a8) |                                                                                                                |
|  Phi-3-instruct  |   `3.0/2.5/2.2`    |             `mini`             |   [`GreenBitAI Phi-3`](https://huggingface.co/collections/GreenBitAI/greenbitai-phi-3-6628d008cdf168398a296c92)   |
|  Qwen-1.5-Chat   | `4.0/3.0/2.5/2.2`  | `0.5B/1.8B/4B/7B/14B/32B/110B` | [`GreenBitAI Qwen 1.5`](https://huggingface.co/collections/GreenBitAI/greenbitai-qwen15-661f86ea69433f3d3062c920) |
|    01-Yi-Chat    |   `3.0/2.5/2.2`    |            `6B/34B`            |   [`GreenBitAI 01-Yi`](https://huggingface.co/collections/GreenBitAI/greenbitai-01-yi-661f88af0648daa766d5102f)   |           

### PPL Evaluation
```bash
python -m green_bit_llm.evaluation.evaluate --model GreenBitAI/Qwen-1.5-4B-layer-mix-bpw-3.0 --trust-remote-code --eval-ppl --ppl-tasks wikitext2,c4,ptb
```

| **Repository** | **Bpw** | **wikitext 2 (2048)** | **c4 (2048)** |
|:--------------:|:-------:|:---------------------:|:-------------:|
|   Llama-3 8B   |   16    |          6.1          |     10.6      |                      
|                |   4.0   |          6.4          |     11.0      |                      
|                |   3.0   |          7.1          |     13.1      |                      
|                |   2.5   |          7.8          |     15.5      |                      
|                |   2.2   |          8.2          |     17.7      |                      
|   Qwen1.5-7B   |   16    |          7.9          |     11.0      |                      
|                |   4.0   |          8.0          |     11.2      |                      
|                |   3.0   |          8.4          |     12.3      |                      
|                |   2.5   |          8.9          |     13.3      |                      
|                |   2.2   |          9.3          |     14.7      |                     
|  Llama-3 70B   |   16    |          2.9          |      8.2      |                      
|                |   4.0   |          3.1          |      6.9      |                      
|                |   3.0   |          4.4          |      8.1      |                      
|                |   2.5   |          5.4          |      9.1      |                      
|                |   2.2   |          5.9          |     10.2      |
 
### Few-Shot Evaluation
```bash
python -m green_bit_llm.evaluation.evaluate --model GreenBitAI/Qwen-1.5-4B-layer-mix-bpw-3.0 --trust-remote-code --batch-size 16 --few-shot-tasks wic,boolq --eval-few-shot
```

The zero-shot evaluation results for low-bit Qwen 1.5 and Llama 3 family are listed as an example:

| **Repository (Qwen Family)**      | **Avg Acc.** | **OpenBQ** | **ARC-E** | **Winogr.** | **HellaS.** | **ARC-C** | **PIQA** | **BoolQ** | **RACE** | **ANLI-R1** | **ANLI-R2** | **ANLI-R3** | **WiC** |
|:----------------------------------|:------------:|:----------:|:---------:|:-----------:|:-----------:|:---------:|:--------:|:---------:|:--------:|:-----------:|:-----------:|:-----------:|:-------:|
| `Qwen-1.5-0.5B-layer-mix-bpw-2.2` |    0.398     |   0.170    |   0.443   |    0.527    |    0.332    |   0.238   |  0.634   |   0.620   |  0.318   |    0.332    |    0.338    |    0.330    |  0.500  | 
| `Qwen-1.5-0.5B-layer-mix-bpw-2.5` |    0.394     |   0.170    |   0.514   |    0.541    |    0.337    |   0.232   |  0.637   |   0.496   |  0.318   |    0.316    |    0.358    |    0.326    |  0.490  |
| `Qwen-1.5-0.5B-layer-mix-bpw-3.0` |    0.407     |   0.198    |   0.533   |    0.536    |    0.348    |   0.234   |  0.671   |   0.552   |  0.323   |    0.330    |    0.333    |    0.335    |  0.495  |
| `Qwen-1.5-1.8B-layer-mix-bpw-2.2` |    0.415     |   0.218    |   0.539   |    0.586    |    0.392    |   0.260   |  0.678   |   0.622   |  0.333   |    0.333    |    0.333    |    0.336    |  0.464  |
| `Qwen-1.5-1.8B-layer-mix-bpw-2.5` |    0.423     |   0.222    |   0.592   |    0.585    |    0.406    |   0.267   |  0.695   |   0.629   |  0.336   |    0.314    |    0.339    |    0.361    |  0.507  |
| `Qwen-1.5-1.8B-layer-mix-bpw-3.0` |    0.438     |   0.246    |   0.576   |    0.563    |    0.413    |   0.277   |  0.694   |   0.645   |  0.352   |    0.323    |    0.336    |    0.343    |  0.492  |
| `Qwen-1.5-4B-layer-mix-bpw-2.2`   |    0.480     |   0.254    |   0.663   |    0.623    |    0.463    |   0.339   |  0.712   |   0.718   |  0.349   |    0.326    |    0.355    |    0.384    |  0.513  |
| `Qwen-1.5-4B-layer-mix-bpw-2.5`   |    0.490     |   0.266    |   0.677   |    0.629    |    0.473    |   0.365   |  0.732   |   0.717   |  0.351   |    0.372    |    0.352    |    0.360    |  0.502  |
| `Qwen-1.5-4B-layer-mix-bpw-3.0`   |    0.502     |   0.268    |   0.678   |    0.642    |    0.494    |   0.358   |  0.755   |   0.757   |  0.380   |    0.395    |    0.395    |    0.392    |  0.519  |
| `Qwen-1.5-7B-layer-mix-bpw-2.2`   |    0.513     |   0.278    |   0.669   |    0.654    |    0.504    |   0.389   |  0.741   |   0.759   |  0.376   |    0.383    |    0.410    |    0.403    |  0.517  |
| `Qwen-1.5-7B-layer-mix-bpw-2.5`   |    0.520     |   0.294    |   0.705   |    0.650    |    0.520    |   0.387   |  0.750   |   0.769   |  0.371   |    0.445    |    0.424    |    0.398    |  0.564  |
| `Qwen-1.5-7B-layer-mix-bpw-3.0`   |    0.531     |   0.292    |   0.713   |    0.654    |    0.545    |   0.405   |  0.764   |   0.807   |  0.383   |    0.424    |    0.393    |    0.414    |  0.627  |
| `Qwen-1.5-14B-layer-mix-bpw-2.5`  |    0.553     |   0.318    |   0.727   |    0.682    |    0.564    |   0.413   |  0.775   |   0.792   |  0.390   |    0.472    |    0.434    |    0.446    |  0.623  |
| `Qwen-1.5-14B-layer-mix-bpw-3.0`  |    0.567     |   0.302    |   0.734   |    0.692    |    0.583    |   0.426   |  0.785   |   0.830   |  0.395   |    0.484    |    0.443    |    0.455    |  0.686  |
| `Qwen-1.5-32B-layer-mix-bpw-3.0`  |    0.599     |   0.346    |   0.775   |    0.722    |    0.620    |   0.492   |  0.807   |   0.853   |  0.444   |    0.515    |    0.494    |    0.478    |  0.642  |

| **Repository (Llama 3 Family)**         | **Avg Acc.** | **OpenBQ** | **ARC-E** | **Winogr.** | **HellaS.** | **ARC-C** | **PIQA** | **BoolQ** | **RACE** | **ANLI-R1** | **ANLI-R2** | **ANLI-R3** | **WiC** |
|:----------------------------------------|:------------:|:----------:|:---------:|:-----------:|:-----------:|:---------:|:--------:|:---------:|:--------:|:-----------:|:-----------:|:-----------:|:-------:|
| `Llama-3-8B-layer-mix-bpw-2.2`          |    0.499     |   0.302    |   0.739   |    0.674    |    0.509    |   0.396   |  0.725   |   0.743   |  0.406   |    0.327    |    0.337    |    0.340    |  0.500  | 
| `Llama-3-8B-layer-mix-bpw-2.5`          |    0.506     |   0.298    |   0.760   |    0.684    |    0.513    |   0.418   |  0.744   |   0.756   |  0.389   |    0.335    |    0.335    |    0.335    |  0.509  |
| `Llama-3-8B-layer-mix-bpw-3.0`          |    0.523     |   0.318    |   0.770   |    0.708    |    0.540    |   0.441   |  0.767   |   0.784   |  0.407   |    0.333    |    0.345    |    0.343    |  0.526  |
| `Llama-3-8B-layer-mix-bpw-4.0`          |    0.542     |   0.338    |   0.791   |    0.729    |    0.591    |   0.484   |  0.797   |   0.799   |  0.398   |    0.337    |    0.345    |    0.352    |  0.545  |
| `Llama-3-8B-instruct-layer-mix-bpw-2.2` |    0.514     |   0.292    |   0.645   |    0.672    |    0.499    |   0.367   |  0.698   |   0.775   |  0.423   |    0.417    |    0.424    |    0.398    |  0.565  |
| `Llama-3-8B-instruct-layer-mix-bpw-2.5` |    0.528     |   0.304    |   0.741   |    0.681    |    0.512    |   0.412   |  0.749   |   0.798   |  0.425   |    0.417    |    0.410    |    0.390    |  0.498  |
| `Llama-3-8B-instruct-layer-mix-bpw-3.0` |    0.547     |   0.316    |   0.787   |    0.690    |    0.530    |   0.459   |  0.768   |   0.800   |  0.437   |    0.435    |    0.417    |    0.387    |  0.548  |
| `Llama-3-8B-instruct-layer-mix-bpw-4.0` |    0.576     |   0.344    |   0.808   |    0.716    |    0.569    |   0.513   |  0.778   |   0.825   |  0.449   |    0.462    |    0.449    |    0.432    |  0.578  |

The 5-shot evaluation results for low-bit Llama 3 models are listed as an example:

| **Repository (Llama 3 Family)**              | **Avg Acc.** | **OpenBQ** |  **ARC-E**  | **Winogr.** | **HellaS.** |  **ARC-C**  |  **PIQA**   | **BoolQ** | **RACE** | **ANLI-R1** | **ANLI-R2** | **ANLI-R3** | **WiC** |
|:---------------------------------------------|:------------:|:----------:|:-----------:|:-----------:|:-----------:|:-----------:|:-----------:|:---------:|:--------:|:-----------:|:-----------:|:-----------:|:-------:|
| `Llama-3-8B-layer-mix-bpw-2.2`               |    0.533     |   0.332    |    0.782    |    0.694    |    0.503    |    0.457    |    0.750    |   0.748   |  0.429   |    0.346    |    0.359    |    0.392    |  0.589  | 
| `Llama-3-8B-layer-mix-bpw-2.5`               |    0.533     |   0.330    |    0.774    |    0.705    |    0.506    |    0.443    |    0.751    |   0.793   |  0.409   |    0.360    |    0.352    |    0.364    |  0.606  | 
| `Llama-3-8B-layer-mix-bpw-3.0`               |    0.548     |   0.354    |    0.792    |    0.711    |    0.541    |    0.471    |    0.774    |   0.804   |  0.423   |    0.394    |    0.370    |    0.395    |  0.551  |
| `Llama-3-8B-layer-mix-bpw-4.0`               |    0.586     |   0.380    |    0.829    |    0.764    |    0.601    |    0.511    |    0.799    |   0.811   |  0.429   |    0.473    |    0.439    |    0.425    |  0.590  | 
| `Llama-3-8B-instruct-layer-mix-bpw-2.2`      |    0.560     |   0.308    |    0.785    |    0.692    |    0.501    |    0.477    |    0.760    |   0.815   |  0.442   |    0.476    |    0.448    |    0.470    |  0.617  | 
| `Llama-3-8B-instruct-layer-mix-bpw-2.5`      |    0.568     |   0.332    |    0.796    |    0.704    |    0.509    |    0.476    |    0.766    |   0.825   |  0.451   |    0.475    |    0.438    |    0.474    |  0.576  | 
| `Llama-3-8B-instruct-layer-mix-bpw-3.0`      |    0.584     |   0.336    |    0.814    |    0.714    |    0.532    |    0.500    |    0.782    |   0.835   |  0.459   |    0.480    |    0.460    |    0.478    |  0.625  | 
| `Llama-3-8B-instruct-layer-mix-bpw-4.0`      |    0.627     |   0.382    |    0.846    |    0.741    |    0.575    |    0.565    |    0.792    |   0.852   |  0.479   |    0.597    |    0.534    |    0.549    |  0.616  | 

