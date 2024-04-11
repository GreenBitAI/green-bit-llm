import os
import argparse
import sys
import json
import pprint

import torch
import torch.nn as nn

from transformers import PreTrainedTokenizer
from tqdm import tqdm

from .lmclass import LMClass
from .utils import create_logger, add_dict_to_json_file, pattern_match
from .datautils import get_loaders

from green_bit_llm.model import load
from pathlib import Path

from lm_eval.api.registry import ALL_TASKS 
from lm_eval import evaluator

import warnings

warnings.filterwarnings('ignore')

# default value for arguments
DEFAULT_MODEL_PATH = "GreenBitAI/Qwen-1.5-1.8B-layer-mix-bpw-2.2"
DEFAULT_SEQLEN = 2048
DEFAULT_RANDOM_SEED = 0
DTYPE = torch.half


@torch.no_grad()
def lm_evaluate(lm, args, logger):
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

    if args.eval_few_shot != "":
        few_shot_tasks = pattern_match(args.few_shot_tasks.split(","), ALL_TASKS)

        eval_results = evaluator.simple_evaluate(
            lm,
            tasks=few_shot_tasks,
            num_fewshot=args.num_fewshot,
            limit = None,
        )
        
        eval_results = eval_results["results"]

        results.update(eval_results)
        logger.info(results)

        # for test of MMLU
        if 'hendrycksTest' in args.few_shot_tasks:
            all_cors = []
            all_cors_norm = []
            subcat_cors = {subcat: [] for subcat_lists in subcategories.values() for subcat in subcat_lists}
            cat_cors = {cat: [] for cat in categories}
            cat_cors_norm = {cat: [] for cat in categories}
            for key in t_results['results'].keys():
                if not 'hendrycksTest' in key:
                    continue
                subject = key.split('-')[-1]
                cors = t_results['results'][key]['acc']
                cors_norm = t_results['results'][key]['acc_norm']
                subcats = subcategories[subject]
                for subcat in subcats:
                    subcat_cors[subcat].append(cors)
                    for key in categories.keys():
                        if subcat in categories[key]:
                            cat_cors[key].append(cors)
                            cat_cors_norm[key].append(cors_norm)
                    all_cors.append(cors)
                    all_cors_norm.append(cors_norm)

            for cat in cat_cors:
                cat_acc = np.mean(cat_cors[cat])
                logger.info("Average accuracy {:.4f} - {}".format(cat_acc, cat))
            weighted_acc = np.mean(all_cors)
            logger.info("Average accuracy: {:.4f}".format(weighted_acc))

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
        help="CUDA device IDs",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Enable trusting remote code for tokenizer",
    )
    parser.add_argument(
        "--use-flash-attention-2",
        action="store_true",
        help="Enable using flash attention v2",
    )
    parser.add_argument(
        "--eos-token",
        type=str,
        default="<|im_end|>",
        help="End of sequence token for tokenizer",
    )
    parser.add_argument(
        "--seqlen",
        type=int,
        default=DEFAULT_SEQLEN,
        help="Sequence length"
    )
    parser.add_argument(
        "--eval-ppl",
        action="store_true",
        help="evaluate LLM prediction perplexity",
    )
    parser.add_argument(
        "--ppl-tasks",
        type=str,
        default="c4_new, ptb",
        help="Specify ppl evaluation task",
    )
    parser.add_argument(
        "--eval-few-shot",
        action="store_true",
        help="Evaluate LLM few-shot learning ability",
    )
    parser.add_argument(
        "--num-fewshot",
        type=int,
        default=0,
        help="Specify num of few shot examples for evaluation",
    )
    parser.add_argument(
        "--few-shot-tasks",
        type=str,
        default="openbookqa,arc_easy,winogrande,hellaswag,arc_challenge,piqa,boolq,race,truthfulqa,anli_r1,anli_r2,anli_r3,wic",
        help="Few-shot learning ability evaluation tasks",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="batch size for few-shot evaluation",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="log/",
        help="Specify save dir for eval results.",
    )
    return parser


def create_device_map(cuda_device_id):
    ids = cuda_device_id.split(',')
    # Create strings in the format "cuda:x" for each ID and put them into the collection
    device_map = {f"cuda:{id}" for id in ids}
    return device_map

def main(args):
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


    lm = LMClass(args.model, batch_size=args.batch_size, config=config, tokenizer=tokenizer, model=model)
    lm.seqlen = args.seqlen
    
    logger = create_logger(Path(args.save_dir))

    eval_results = lm_evaluate(lm=lm, args=args, logger=logger)

    eval_results = {"{}".format(args.model): eval_results}

    add_dict_to_json_file(file_path="{}".format(args.save_dir + "eval_results.json"), new_data=eval_results)

if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("Warning: CUDA is needed to run the model.")
        sys.exit(0)

    parser = setup_arg_parser()
    args = parser.parse_args()

    main(args)
