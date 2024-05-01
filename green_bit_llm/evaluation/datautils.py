from transformers import AutoTokenizer
from datasets import load_dataset
import torch
import random

from colorama import init, Fore, Style
init(autoreset=True)

def get_wikitext2(nsamples, seed, seqlen, model):
    """
    Prepares data loaders for the Wikitext-2 dataset for training and testing.

    Args:
        nsamples (int): Number of random samples to generate from the training data.
        seed (int): Seed for random number generator to ensure reproducibility.
        seqlen (int): Sequence length for each input sample.
        model (str): Pretrained model identifier used for tokenization.

    Returns:
        tuple: A tuple containing the training loader and tokenized test data.
    """
    print(Style.BRIGHT + Fore.CYAN + "Info: get_wikitext2")
    traindata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train', cache_dir="./cache/")
    testdata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test', cache_dir="./cache/")

    tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False, trust_remote_code=True)
    trainenc = tokenizer("\n\n".join(traindata['text']), return_tensors='pt')
    testenc = tokenizer("\n\n".join(testdata['text']), return_tensors='pt')

    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))

    return trainloader, testenc


def get_ptb(nsamples, seed, seqlen, model):
    """
    Load and prepare the Penn Treebank (PTB) dataset for training and validation.

    Args:
        nsamples (int): The number of samples to generate for the training loader.
        seed (int): The seed value for random number generation, ensuring reproducibility.
        seqlen (int): The sequence length of each sample.
        model (str): The model identifier used to load a pre-trained tokenizer.

    Returns:
        tuple: A tuple containing the training loader and tokenized validation data.
    """
    print(Style.BRIGHT + Fore.CYAN + "Info: get_ptb")
    traindata = load_dataset('ptb_text_only', 'penn_treebank', split='train', cache_dir="./cache/")
    valdata = load_dataset('ptb_text_only', 'penn_treebank', split='validation', cache_dir="./cache/")

    tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False, trust_remote_code=True)

    trainenc = tokenizer("\n\n".join(traindata['sentence']), return_tensors='pt')
    testenc = tokenizer("\n\n".join(valdata['sentence']), return_tensors='pt')

    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))

    return trainloader, testenc


def get_c4(nsamples, seed, seqlen, model):
    """
    Loads and preprocesses the C4 dataset for training and validation.
    Args:
        nsamples (int): Number of samples to generate for training.
        seed (int): Random seed for reproducibility.
        seqlen (int): The sequence length for each training sample.
        model (str): Model identifier for tokenizer initialization.

    Returns:
        tuple: A tuple containing training data loader and validation data tensor.
    """
    print(Style.BRIGHT + Fore.CYAN + "Info: get_c4")
    traindata = load_dataset(
        'allenai/c4', data_files={'train': 'en/c4-train.00000-of-01024.json.gz'},
        split='train',
        cache_dir="./cache/"
    )
    valdata = load_dataset(
        'allenai/c4', data_files={'validation': 'en/c4-validation.00000-of-00008.json.gz'},
        split='validation',
        cache_dir="./cache/"
    )

    tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False, trust_remote_code=True)

    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):

        while True:
            i = random.randint(0, len(traindata) - 1)
            trainenc = tokenizer(traindata[i]['text'], return_tensors='pt')
            if trainenc.input_ids.shape[1] > seqlen + 2:
                break

        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))

    random.seed(0)
    valenc = []
    for _ in range(256):
        while True:
            i = random.randint(0, len(valdata) - 1)
            tmp = tokenizer(valdata[i]['text'], return_tensors='pt')
            if tmp.input_ids.shape[1] >= seqlen:
                break
        i = random.randint(0, tmp.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        valenc.append(tmp.input_ids[:, i:j])
    valenc = torch.hstack(valenc)

    return trainloader, valenc


def get_ptb_new(nsamples, seed, seqlen, model):
    """
    Generates training and testing data loaders for the Penn Treebank dataset using a specified model tokenizer.

    Args:
        nsamples (int): Number of samples to generate in the training loader.
        seed (int): Random seed for reproducibility of sample selection.
        seqlen (int): Sequence length of each sample in the training data.
        model (str): Model identifier for the tokenizer (e.g., a Hugging Face model name).

    Returns:
        tuple: A tuple containing the training loader (list of tuples with input IDs and target IDs) and
               the tokenized test data.
    """
    print(Style.BRIGHT + Fore.CYAN + "Info: get_ptb_new")
    traindata = load_dataset('ptb_text_only', 'penn_treebank', split='train')
    testdata = load_dataset('ptb_text_only', 'penn_treebank', split='test')

    tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False, trust_remote_code=True)

    trainenc = tokenizer(" ".join(traindata["sentence"]), return_tensors="pt")
    testenc = tokenizer(" ".join(testdata["sentence"]), return_tensors="pt")

    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))

    return trainloader, testenc


def get_c4_new(nsamples, seed, seqlen, model):
    """
    Load and preprocess training and validation datasets from C4 dataset, and tokenize the data.

    Args:
    nsamples (int): Number of samples to process for the training data.
    seed (int): Random seed for reproducibility of sample selection.
    seqlen (int): Length of the sequence for each input/output example.
    model (str): Model identifier for the tokenizer, specifying which pretrained model to use.

    Returns:
    tuple: A tuple containing two elements:
           - trainloader (list of tuples): A list where each tuple contains input ids and target tensors for training.
           - valenc (torch.Tensor): A tensor containing the tokenized validation data.
    """
    print(Style.BRIGHT + Fore.CYAN + "Info: get_c4_new")
    traindata = load_dataset(
        'allenai/c4', data_files={'train': 'en/c4-train.00000-of-01024.json.gz'}, split='train'
    )
    valdata = load_dataset(
        'allenai/c4', data_files={'validation': 'en/c4-validation.00000-of-00008.json.gz'}, split='validation'
    )

    tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False, trust_remote_code=True)

    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        while True:
            i = random.randint(0, len(traindata) - 1)
            trainenc = tokenizer(traindata[i]["text"], return_tensors="pt")
            if trainenc.input_ids.shape[1] >= seqlen:
                break
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))

    valenc = tokenizer(" ".join(valdata[:1100]["text"]), return_tensors="pt")
    valenc = valenc.input_ids[:, : (256 * seqlen)]

    return trainloader, valenc


def get_loaders(name, nsamples=128, seed=0, seqlen=2048, model=''):
    """
    Retrieves data loaders for different datasets based on the dataset name.

    Args:
        name (str): The name of the dataset to load, which can include specific versions like 'new'.
        nsamples (int): The number of samples to retrieve from the dataset.
        seed (int): The random seed to ensure reproducibility.
        seqlen (int): The sequence length of the samples.
        model (str): The model specification that might influence data preprocessing.

    Returns:
        DataLoader: A configured data loader for the specified dataset.

    Raises:
        ValueError: If the dataset name is not recognized or supported.
    """
    if 'wikitext2' in name:
        return get_wikitext2(nsamples, seed, seqlen, model)
    elif 'ptb' in name:
        if 'new' in name:
            return get_ptb_new(nsamples, seed, seqlen, model)
        else:
            return get_ptb(nsamples, seed, seqlen, model)
    elif 'c4' in name:
        if 'new' in name:
            return get_c4_new(nsamples, seed, seqlen, model)
        else:
            return get_c4(nsamples, seed, seqlen, model)
    else:
        raise ValueError(f"Only support wikitext2, c4, c4_new, ptb, ptb_new currently, but get {name}")
