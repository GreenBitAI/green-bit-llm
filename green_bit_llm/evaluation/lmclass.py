import sys
from tqdm import tqdm
from typing import List, Optional, Tuple, Union
from lm_eval.api.model import LM
from lm_eval.api.instance import Instance
from lm_eval.models.huggingface import HFLM
from lm_eval import utils
import lm_eval.models.utils
import torch.nn.functional as F
import torch
from lm_eval.models.utils import (
    Collator,
    clear_torch_cache,
    get_dtype,
    pad_and_concat,
    stop_sequences_criteria,
)

from colorama import init, Fore, Style
init(autoreset=True)

class LMClass(LM):
    """
    Wraps a pretrained language model into a format suitable for evaluation tasks.
    This class adapts a given model to be used with specific language modeling evaluation tools.

    Args:
        model_name (str): Name of the language model.
        batch_size (int): Batch size per GPU for processing.
        config (dict): Configuration settings for the model.
        tokenizer: Tokenizer associated with the model for text processing.
        model (nn.Module): The pretrained neural network model.
    """
    def __init__(self, model_name, batch_size, config, tokenizer, model):
        # Initializes the model wrapper class with specified model and configuration
        super().__init__()

        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        self.batch_size_per_gpu = batch_size

        self.model_config = config
        self.tokenizer = tokenizer
        self.model = model
        #self._model = model
        self.initial()
    
    def initial(self):
        # Initializes the model for inference, setting up necessary parameters such as sequence length and vocab size
        self.seqlen = self.model.config.max_position_embeddings
        self.model.eval()
        self.vocab_size = self.tokenizer.vocab_size
        print(Style.BRIGHT + Fore.CYAN + "Info: vocab size: ", self.vocab_size)

    @property
    def eot_token(self) -> str:
        # Returns the end-of-text token as a string
        return self.tokenizer.eos_token

    @property
    def eot_token_id(self):
        # we use EOT because end of *text* is more accurate for what we're doing than end of *sentence*
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        # Returns the maximum length of sequences the model can handle, based on the model's configuration
        try:
            return self.gpt2.config.n_ctx
        except AttributeError:
            # gptneoconfig doesn't have n_ctx apparently
            return self.model.config.max_position_embeddings

    @property
    def max_gen_toks(self):
        # Returns the maximum number of tokens that can be generated in one go
        print(Style.BRIGHT + Fore.CYAN + "Info: max_gen_toks fn")
        return 256

    @property
    def batch_size(self):
        # Returns the configured batch size per GPU
        # TODO: fix multi-gpu
        return self.batch_size_per_gpu  # * gpus

    @property
    def device(self):
        # Returns the computing device (CPU or GPU) that the model is using
        # TODO: fix multi-gpu
        return self._device

    def tok_encode(self, string: str):
        # Encodes a string into its corresponding IDs using the tokenizer
        return self.tokenizer.encode(string, add_special_tokens=False)

    def tok_encode_batch(self, strings):
        # Encodes a batch of strings into model inputs, handling padding and special tokens
        return self.tokenizer(
            strings,
            padding=True,
            add_special_tokens=False,
            return_tensors="pt",
        )

    def tok_decode(self, tokens):
        # Decodes a batch of token IDs back into strings
        return self.tokenizer.batch_decode(tokens, skip_special_tokens=True)

    def _model_call(self, inps):
        """
        Performs a forward pass through the model with the provided inputs and returns logits

        Args:
            inps: a torch tensor of shape [batch, sequence]
            the size of sequence may vary from call to call
            returns: a torch tensor of shape [batch, sequence, vocab] with the
            logits returned from the model
        """
        with torch.no_grad():
            return self.model(inps)["logits"]

    def model_batched_set(self, inps):
        # Processes a set of inputs in batches and returns a list of logit tensors
        dataset_logits = []
        for batch in inps:
            multi_logits = F.log_softmax(
                self._model_call(batch), dim=-1
            ).cpu()  # [batch, padding_length, vocab]
            dataset_logits.append(multi_logits)
        return dataset_logits

    def _model_generate(self, context, max_length, eos_token_id):
        # Generates text based on a given context up to a specified maximum length
        return self.model.generate(
            context, max_length=max_length, eos_token_id=eos_token_id, do_sample=False
        )
    
    def _batch_scheduler(self, pos, n_reordered_requests):
        sched = pos // int(len(n_reordered_requests) / self.batch_schedule)
        if sched in self.batch_sizes:
            return self.batch_sizes[sched]
        if (len(self.batch_sizes) > 1) and (
            self.batch_sizes[sched - 1] == self.max_batch_size
        ):
            # if previous batch size is already maximal, skip recomputation
            self.batch_sizes[sched] = self.max_batch_size
            return self.batch_sizes[sched]
        print(
            f"Passed argument batch_size = auto:{self.batch_schedule}. Detecting largest batch size"
        )
        self.batch_sizes[sched] = self._detect_batch_size(n_reordered_requests, pos)
        print(f"Determined largest batch size: {self.batch_sizes[sched]}")
        return self.batch_sizes[sched]

    def loglikelihood(self, requests) -> List[Tuple[float, bool]]:
        """
        Copied directly from
        https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/models/huggingface.py
        """
        new_reqs = []
        for context, continuation in [req.args for req in requests]:
            if context == "":
                raise NotImplementedError(
                    "Implementing empty context is not supported yet"
                )
            context_enc, continuation_enc = self._encode_pair(context, continuation)

            new_reqs.append(((context, continuation), context_enc, continuation_enc))

        return self._loglikelihood_tokens(new_reqs)

    def _loglikelihood_tokens(
        self,
        requests: List[Tuple[Tuple[str, str], List[int], List[int]]],
        disable_tqdm: bool = False,
    ) -> List[Tuple[float, bool]]:
        """
        The function to compute the loglikelihood of the continuation
        tokens given the context tokens.

        This function is an adapted version of the original function from
        https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/models/huggingface.py
        """
        res = []

        def _collate(req: Tuple[Tuple[str, str], List[int], List[int]]):
            """Defines the key for the sorted method"""
            # the negative sign on len(toks) sorts descending - this has a few advantages:
            # - time estimates will always be over not underestimates, which is more useful for planning
            # - to know the size of a batch when going through the list, you know the first one is always the batch
            #   padded context length. this is useful to simplify the batching logic and more importantly to make
            #   automatic adaptive batches much much easier to implement
            # - any OOMs will happen right away rather than near the end

            toks = req[1] + req[2]
            return -len(toks), tuple(toks)

        def _lookup_one_token_cont(req: Tuple[Tuple[str, str], List[int], List[int]]):
            """Defines the key to group and lookup one-token continuations"""
            # Use with group_by="contexts" (optional)"
            # allows for the creation of a lookup, so we can reuse logits in case of one-token continuations.
            # speeds up some multiple-choice tasks proportionally to the number of choices.
            # groups requests by context+continuation[:-1] and infer on one request/group.
            return req[-2] + req[-1][:-1]

        #re_ord = utils.Reorderer(requests, _collate)
        re_ord = Collator(
            requests,
            sort_fn=_collate,
            group_by="contexts",
            group_fn=_lookup_one_token_cont,
        )
        
        # automatic (variable) batch size detection for vectorization
        # pull longest context sample from request
        n_reordered_requests = len(re_ord)
        batch_size = (
            self.batch_size
            if self.batch_size != "auto"
            else 0
        )
        '''
        batch_fn = (
            self._batch_scheduler
            if self.batch_size == "auto"
            else None
        )
        '''
        batch_fn = None

        chunks = re_ord.get_batched(n=batch_size, batch_fn=batch_fn)
        pbar = tqdm(
            total=len(requests),
            disable=(disable_tqdm or (self.rank != 0)),
            desc="Running loglikelihood requests",
        )
        for chunk in chunks:
            batch_inp = []
            cont_toks_list = []
            inplens = []
            
            conts = []
            encoder_attns = []
            
            padding_len_inp = None
            padding_len_cont = None

            # len(chunk) is the batch_size
            for cache_key, context_enc, continuation_enc in chunk:
                # how this all works (illustrated on a causal decoder-only setup):
                #          CTX      CONT
                # inp    0 1 2 3|4 5 6 7 8 9   <- last token is deleted by inp[:, :-1]
                # model  \               \
                # logits   1 2 3|4 5 6 7 8 9   <- the ctx half gets tossed out by the
                # cont_toks      4 5 6 7 8 9      [:, -len(continuation_enc):, :self.vocab_size] slice # noqa: E501

                inp = torch.tensor((context_enc + continuation_enc)[-(self.max_length + 1) :][:-1],
                        dtype=torch.long,
                        device=self.device)

                (inplen,) = inp.shape
                
                padding_len_inp = (
                    max(padding_len_inp, inplen)
                    if padding_len_inp is not None
                    else inplen
                )

                batch_inp.append(inp)
                cont_toks_list.append(continuation_enc)
                inplens.append(inplen)
            
            #print(padding_len_inp)
            #print(batch_inp)

            batch_inp = pad_and_concat(
                    padding_len_inp, batch_inp, padding_side="right")  # [batch, padding_len_inp]

            multi_logits = F.log_softmax(self.model(input_ids=batch_inp).logits, dim=-1)
            
            for (request_str, ctx_tokens, _), logits, inplen, cont_toks in zip(chunk, multi_logits, inplens, cont_toks_list):
                # Slice to original seq length
                contlen = len(cont_toks)
                # take only logits in the continuation
                # (discard context toks if decoder-only ; discard right-padding)
                # also discards + checks for "virtual tokens" in the causal LM's input window
                # from prompt/prefix tuning tokens, if applicable
                ctx_len = (
                    inplen + (logits.shape[0] - padding_len_inp)
                )
                logits = self._select_cont_toks(logits, contlen=contlen, inplen=ctx_len)
                logits = logits.unsqueeze(0)  # [1, seq, vocab]

                # Check if per-token argmax is exactly equal to continuation
                greedy_tokens = logits.argmax(dim=-1)

                # check for one-token continuation cache hits.
                # noop in case group_by != "contexts" or no cache hit and returns the
                # original args. Otherwise, expands the logits batch dimension and yields each
                # batch along with matching continuation tokens and prompt strings.
                # logits -> [1, seq, vocab]
                for request_str, cont_toks, logits in re_ord.get_cache(
                    req_str=request_str,
                    cxt_toks=ctx_tokens,
                    cont_toks=cont_toks,
                    logits=logits,):

                    cont_toks = torch.tensor(
                        cont_toks, dtype=torch.long, device=self.device
                    ).unsqueeze(0)  # [1, seq]
                    max_equal = (greedy_tokens == cont_toks).all()

                    # Obtain log-probs at the corresponding continuation token indices
                    # last_token_slice = logits[:, -1, :].squeeze(0).tolist()
                    logits = torch.gather(logits, 2, cont_toks.unsqueeze(-1)).squeeze(
                        -1
                    )  # [1, seq]

                    # Answer: (log prob, is-exact-match)
                    answer = (float(logits.sum()), bool(max_equal))

                    res.append(answer)

                    self.cache_hook.add_partial("loglikelihood", request_str, answer)
                    pbar.update(1)
        
        pbar.close()

        return re_ord.get_original(res)

    def loglikelihood_rolling(self, requests: List[Instance]) -> List[float]:
        raise NotImplementedError(
            "The method not required by any of our current task integrations so far"
        )

    def generate_until(self, requests: List[Instance]) -> List[str]:
        """
        The function to generate a certain number of new tokens
        given a context.

        This function is an adapted version of the original function from
        https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/models/openai_completions.py
        """
        raise NotImplementedError(
            "The method not required by any of our current evaluation task integrations so far, will get updated soon"
        )

        if not requests:
            return []
        res = []
        requests = [req.args for req in requests]

        def _collate(x):
            toks = self.tok_encode(x[0])
            return len(toks), x[0]

        re_ord = utils.Reorderer(requests, _collate)

        def sameuntil_chunks(xs, size):
            ret = []
            lastuntil = xs[0][1]
            for x in xs:
                if len(ret) >= size or x[1] != lastuntil:
                    yield ret, lastuntil
                    ret = []
                    lastuntil = x[1]
                ret.append(x)

            if ret:
                yield ret, lastuntil

        pbar = tqdm(total=len(requests))
        for chunk, request_args in tqdm(
            list(sameuntil_chunks(re_ord.get_reordered(), self.batch_size))
        ):
            inps = []

            # make a deepcopy since we are changing arguments
            request_args = copy.deepcopy(request_args)

            self._max_gen_toks = request_args.pop("max_gen_toks", self.max_gen_toks)

            for context, _ in chunk:
                # add context (prompts) to the list
                inps.append(context)

            until = request_args.pop("until", ["<|endoftext|>"])
            request_args.pop("do_sample", None)
            request_args["temperature"] = request_args.get("temperature", 0)

            # run inference (generate max_gen_toks tokens)
            out = self.model(
                sequences=inps,
                max_new_tokens=self.max_gen_toks - 1,
                stop=until,
                **request_args,
            )

            for resp, (context, args_) in zip(out.generations, chunk):
                text = resp.text
                until_ = until
                # split the text at the first occurrence of any of the until tokens
                for term in until_:
                    if len(term) > 0:
                        text = text.split(term)[0]

                res.append(text)

                self.cache_hook.add_partial(
                    "generate_until", (context, {"until": until_}), text
                )
                pbar.update(1)

        pbar.close()

        return re_ord.get_original(res)
    
    def _select_cont_toks(
        self, logits: torch.Tensor, contlen: int = None, inplen: int = None
    ) -> torch.Tensor:
        assert (contlen and inplen), "Must pass input len and cont. len to select scored logits for causal LM"
        # discard right-padding.
        # also discard the input/context tokens. we'll only score continuations.
        logits = logits[inplen - contlen : inplen]
        return logits

    def _encode_pair(
        self, context: str, continuation: str
    ) -> Tuple[List[int], List[int]]:
        """
        Copied directly from
        https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/models/huggingface.py
        """
        n_spaces = len(context) - len(context.rstrip())
        if n_spaces > 0:
            continuation = context[-n_spaces:] + continuation
            context = context[:-n_spaces]
        whole_enc = self.tok_encode(context + continuation)
        context_enc = self.tok_encode(context)
        context_enc_len = len(context_enc)
        continuation_enc = whole_enc[context_enc_len:]
        return context_enc, continuation_enc
