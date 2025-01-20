import os
import argparse
from pathlib import Path
from tqdm import tqdm
import transformers
from transformers.data import metrics
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from vllm.model_executor.layers.logits_processor import _apply_logits_processors
import torch
import torch.nn as nn
from green_bit_llm.evaluation.datautils import get_loaders
import warnings
import json

warnings.filterwarnings('ignore')

# default values
DEFAULT_MODEL_PATH = "/workspace/models/Qwen2.5-7B-Instruct"
DEFAULT_SEQLEN = 2048
DEFAULT_RANDOM_SEED = 0

logits_list = []
def forward_hook(module, input, output):
    lm_head, hidden_states, sampling_metadata, *embedding_bias = input
    embedding_bias = embedding_bias[0] if embedding_bias else None
    logits = module._get_logits(hidden_states, lm_head, embedding_bias)
    if logits is not None:
        if module.soft_cap is not None:
            logits = logits / module.soft_cap
            logits = torch.tanh(logits)
            logits = logits * module.soft_cap
        if module.scale != 1.0:
            logits *= module.scale
        logits = _apply_logits_processors(logits, sampling_metadata)
        logits_list.append(logits)
    return output 

@torch.no_grad()
def calculate_ppl(model, testenc, seqlen, device='cuda'):
    nsamples = testenc.numel() // seqlen
    nlls = []
    
    sampling_params = SamplingParams(
        temperature=1.0,
        max_tokens=1,
        logprobs=None
    )
    
    for i in tqdm(range(nsamples)):
        logits_list.clear()
        batch = testenc[:, (i * seqlen):((i + 1) * seqlen)]
        outputs = model.generate(prompts=None, prompt_token_ids=batch.tolist(), sampling_params=sampling_params)
        logits = logits_list[0].to(device)
        logits = logits.unsqueeze(0)
        shift_logits = logits[:, :-1, :]
        shift_labels = testenc[:, (i * seqlen): ((i + 1) * seqlen)][
                        :, 1:
                        ].to(device)
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        )
        neg_log_likelihood = loss.float() * seqlen
        nlls.append(neg_log_likelihood)
    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * seqlen))
    return ppl.item()

def setup_arg_parser():
    """设置参数解析器"""
    parser = argparse.ArgumentParser(description="VLLM evaluate script")
    parser.add_argument("--seed", type=int, default=DEFAULT_RANDOM_SEED)
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--seqlen", type=int, default=DEFAULT_SEQLEN)
    parser.add_argument("--ppl-tasks", type=str, default="wikitext2,c4_new,ptb")
    parser.add_argument("--save-dir", type=str, default="log/")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.5)
    return parser

def main():
    if not torch.cuda.is_available():
        print("Warning: CUDA is needed to run the model.")
        sys.exit(0)
        
    parser = setup_arg_parser()
    args = parser.parse_args()
    
    if not os.path.exists(Path(args.save_dir)):
        os.makedirs(Path(args.save_dir))
    
    print(f"Loading model from {args.model}")
    model = LLM(
        model=args.model,
        trust_remote_code=True,
        gpu_memory_utilization=args.gpu_memory_utilization
    )
    model.llm_engine.model_executor.driver_worker.model_runner.model.logits_processor.register_forward_hook(forward_hook)
        
    results = {}
    
    for dataset in args.ppl_tasks.split(","):
        print(f"\nEvaluating {dataset}...")
        dataloader, testloader = get_loaders(
            dataset.strip(),
            seed=args.seed,
            model=args.model,
            seqlen=args.seqlen,
        )
        
        if "c4" in dataset:
            testenc = testloader
        else:
            testenc = testloader.input_ids
 
        ppl = calculate_ppl(model, testenc, args.seqlen)
        print(f"{dataset} PPL: {ppl}")
        results[dataset] = ppl
    
    eval_results = {args.model: results}
    result_path = os.path.join(args.save_dir, "vllm_eval_results.json")
    
    with open(result_path, 'w') as f:
        json.dump(eval_results, f, indent=4)
    
    print(f"\nResults saved to {result_path}")

if __name__ == "__main__":
    main()