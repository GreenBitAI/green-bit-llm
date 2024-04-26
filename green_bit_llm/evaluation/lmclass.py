from lm_eval.base import BaseLM
import torch.nn.functional as F
import torch

from colorama import init, Fore, Style
init(autoreset=True)

class LMClass(BaseLM):
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
