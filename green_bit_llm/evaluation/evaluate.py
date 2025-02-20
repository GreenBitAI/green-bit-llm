import os
import argparse
import sys

import torch
import torch.nn as nn

from tqdm import tqdm

import safetensors
from peft import PeftModel, LoraConfig, get_peft_model

from .lmclass import LMClass
from .utils import create_logger, add_dict_to_json_file
from .datautils import get_loaders

from green_bit_llm.common import load
from green_bit_llm.common.utils import get_model_path
from green_bit_llm.sft.peft_utils.model import *
from peft import PeftModel, LoraConfig, get_peft_model
from pathlib import Path
 
from lm_eval import evaluator
from vllm.model_executor.layers.logits_processor import _apply_logits_processors
from vllm import LLM, SamplingParams
import warnings

warnings.filterwarnings('ignore')



# default value for arguments
DEFAULT_MODEL_PATH = "GreenBitAI/Qwen-1.5-1.8B-layer-mix-bpw-2.2"
DEFAULT_SEQLEN = 2048
DEFAULT_RANDOM_SEED = 0
DTYPE = torch.half
DEFAULT_MODEL_BCKEND = ["vllm", "greenbit-engine"]

replace_peft_lora_model_with_gba_lora_model()

@torch.no_grad()
def lm_evaluate(lm, args, logger):
    """
    Evaluates the language model (lm) according to the specified evaluation parameters in args.
    This function handles both perplexity evaluation on various datasets and few-shot learning evaluation.

    Parameters:
        lm: The language model object configured for evaluation.
        args: An object containing all necessary parameters and flags for evaluation, such as which datasets to use,
              whether to evaluate perplexity or few-shot performance, sequence length, etc.
        logger: A logging object used to record evaluation results and other important messages.

    Returns:
        A dictionary containing evaluation results, including perplexity values and few-shot evaluation metrics,
        keyed by dataset or task name.
    """
    results = {}
    lm.model = lm.model.to(lm.device)

    if args.eval_ppl:
        for dataset in args.ppl_tasks.split(","):
            dataloader, testloader = get_loaders(
                dataset,
                seed=args.seed,
                model=args.model,
                seqlen=args.seqlen,
            )
            
            if "c4" in dataset:
                testenc = testloader
            else:
                testenc = testloader.input_ids

            nsamples = testenc.numel() // lm.seqlen
            use_cache = lm.model.config.use_cache
            lm.model.config.use_cache = False
            lm.model.eval()
            nlls = []
            
            for i in tqdm(range(nsamples)):
                batch = testenc[:, (i * lm.seqlen): ((i + 1) * lm.seqlen)].to(lm.device)
                with torch.no_grad():
                    outputs = lm.model.model(batch)
                hidden_states = outputs[0]
                logits = lm.model.lm_head(hidden_states)
                shift_logits = logits[:, :-1, :]
                shift_labels = testenc[:, (i * lm.seqlen): ((i + 1) * lm.seqlen)][
                               :, 1:
                               ].to(lm.model.lm_head.weight.device)
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                )
                neg_log_likelihood = loss.float() * lm.seqlen
                nlls.append(neg_log_likelihood)
                
            ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * lm.seqlen))
            logger.info(f'{dataset} : {ppl.item()}')
            lm.model.config.use_cache = use_cache
            results[dataset] = ppl.item()

    if args.eval_few_shot and args.eval_few_shot != "":
        few_shot_tasks = args.few_shot_tasks.split(",")

        eval_results = evaluator.simple_evaluate(
            lm,
            tasks=few_shot_tasks,
            batch_size=args.batch_size,
            num_fewshot=args.num_fewshot,
            no_cache=True
        )
        
        results.update({"results": eval_results["results"]})
        results.update({"versions": eval_results["versions"]})
        logger.info(evaluator.make_table(results))

    return results

def setup_arg_parser():
    """Set up and return the argument parser."""
    parser = argparse.ArgumentParser(description="green-bit-llm evaluate script")
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_RANDOM_SEED,
        help="The random seed for data loader.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL_PATH,
        help="The path to the local model directory or Hugging Face repo.",
    )
    parser.add_argument(
        "--cuda-device-id",
        type=str,
        default="0",
        help="CUDA device IDs.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Enable trusting remote code for tokenizer.",
    )
    parser.add_argument(
        "--use-flash-attention-2",
        action="store_true",
        help="Enable using flash attention v2.",
    )
    parser.add_argument(
        "--eos-token",
        type=str,
        default="<|im_end|>",
        help="End of sequence token for tokenizer.",
    )
    parser.add_argument(
        "--seqlen",
        type=int,
        default=DEFAULT_SEQLEN,
        help="Sequence length."
    )
    parser.add_argument(
        "--eval-ppl",
        action="store_true",
        help="Evaluate LLM prediction perplexity.",
    )
    parser.add_argument(
        "--ppl-tasks",
        type=str,
        default="wikitext2, c4_new, ptb",
        help="Specify ppl evaluation task.",
    )
    parser.add_argument(
        "--eval-few-shot",
        action="store_true",
        help="Evaluate LLM few-shot learning ability.",
    )
    parser.add_argument(
        "--num-fewshot",
        type=int,
        default=0,
        help="Specify num of few shot examples for evaluation.",
    )
    parser.add_argument(
        "--few-shot-tasks",
        type=str,
        default="openbookqa,arc_easy,winogrande,hellaswag,arc_challenge,piqa,boolq,race,anli_r1,anli_r2,anli_r3,wic",
        help="Few-shot learning ability evaluation tasks.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for few-shot evaluation.",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="log/",
        help="Specify save dir for eval results.",
    )
    parser.add_argument(
        "--lora-dir",
        type=str,
        default=None,
        help="Specify lora dir for lora merge"

    )
    parser.add_argument(
        "--backend",
        type=str,
        default="vllm",
        help="Specify the model inference backend from [vllm, greenbit-engine]"
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.8,
        help="only useful when using vllm backend."
    )
    return parser


def create_device_map(cuda_device_id):
    ids = cuda_device_id.split(',')
    # Create strings in the format "cuda:x" for each ID and put them into the collection
    device_map = {f"cuda:{id}" for id in ids}
    return device_map

def evaluate_green_bit_engine(args):
    if not os.path.exists(Path(args.save_dir)):
        os.mkdir(Path(args.save_dir))
        
    # Building configs
    tokenizer_config = {"trust_remote_code": True if args.trust_remote_code else None}
    pretrain_model_config = {
        "trust_remote_code": True if args.trust_remote_code else None,
        "use_flash_attention_2": True if args.use_flash_attention_2 else None
    }

    if args.eos_token is not None:
        tokenizer_config["eos_token"] = args.eos_token

    model, tokenizer, config = load(
        args.model,
        tokenizer_config=tokenizer_config,
        dtype=DTYPE,
        device_map='auto',
        seqlen=args.seqlen,
        model_config=pretrain_model_config,
        requires_grad=False
    )

    if args.lora_dir is not None:
        config = LoraConfig(
            r=64,
            lora_alpha=32,
            target_modules=["q_proj", "v_proj", "out_proj", "up_proj"],
            lora_dropout=0.01,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model,config)
        model.load_adapter(args.lora_dir, adapter_name="default")

    lm = LMClass(args.model, batch_size=args.batch_size, config=config, tokenizer=tokenizer, model=model)
    lm.seqlen = args.seqlen
    
    logger = create_logger(Path(args.save_dir))
    
    with torch.no_grad():
        eval_results = lm_evaluate(lm=lm, args=args, logger=logger)

    eval_results = {"{}".format(args.model): eval_results}

    add_dict_to_json_file(file_path="{}".format(os.path.join(args.save_dir, "eval_greenbit_engine_results.json")), new_data=eval_results)

def evaluate_vllm(args):
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
    
    print(f"Loading model from {args.model}")
    model = LLM(
        model=args.model,
        trust_remote_code=args.trust_remote_code,
        gpu_memory_utilization=args.gpu_memory_utilization
    )
    model.llm_engine.model_executor.driver_worker.model_runner.model.logits_processor.register_forward_hook(forward_hook)
        
    results = {}
    logger = create_logger(Path(args.save_dir))
    if args.eval_ppl:
        for dataset in args.ppl_tasks.split(","):
            # print(f"\nEvaluating {dataset}...")
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
            logger.info(f'{dataset} : {ppl}')
            results[dataset] = ppl

    eval_results = {args.model: results}
    
    add_dict_to_json_file(file_path="{}".format(os.path.join(args.save_dir, "eval_vllm_results.json")), new_data=eval_results)
    
def main(args):
    if args.backend not in DEFAULT_MODEL_BCKEND:
        print(f"Backend is error, please set the backend from {DEFAULT_MODEL_BCKEND}")
        exit(-1)
    if args.backend == "vllm":
        evaluate_vllm(args)
    elif args.backend == "greenbit-engine":
        evaluate_green_bit_engine(args)

if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("Warning: CUDA is needed to run the model.")
        sys.exit(0)

    parser = setup_arg_parser()
    args = parser.parse_args()

    main(args)
