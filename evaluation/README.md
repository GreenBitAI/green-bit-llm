## Evaluation

Example evaluation scripts for the GreenBitAI family of low-bit models.
- `evaluate.py` contains code to evaluate model ppl performance on `wikitext2,c4_new,ptb` tasks, and few shot ability using `lm_eval` library.
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

### PPL Evaluation
```bash
python -m evaluation.evaluate --model GreenBitAI/Qwen-1.5-4B-layer-mix-bpw-3.0 --trust-remote-code --eval-ppl -ppl-tasks wikitext2,c4_new,ptb
```

### Few-Shot Evaluation
```bash
python -m evaluation.evaluate --model GreenBitAI/Qwen-1.5-4B-layer-mix-bpw-3.0 --trust-remote-code --batch-size 16 --few-shot-tasks wic,boolq --eval-few-shot
```
