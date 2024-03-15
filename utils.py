import os
import torch
import random
import logging
import datasets
import subprocess
import numpy as np
import transformers
import torch.nn as nn
import torch.nn.functional as F
from external import GradualWarmupScheduler
from datasets import load_dataset, DownloadConfig
from transformers import (
    AutoModelForCausalLM, AutoTokenizer,
    AutoModelForSequenceClassification, AutoConfig, get_scheduler
)


def git_status():
    process = subprocess.Popen(['git', 'rev-parse', 'HEAD'],
                               shell=False,
                               stdout=subprocess.PIPE)
    git_head_hash = process.communicate()[0].strip().decode("utf-8")
    git_head_hash_short = git_head_hash[:7]

    process = subprocess.Popen(['git', 'rev-parse', '--abbrev-ref', 'HEAD'],
                               shell=False,
                               stdout=subprocess.PIPE)
    git_branch_name = process.communicate()[0].strip().decode("utf-8")

    return git_branch_name, git_head_hash_short


def set_log(log_file=None):
    log_level = "info"  # TODO: avoid hard-coding
    log_level = {
        'critical': logging.CRITICAL,
        'error': logging.ERROR,
        'warn': logging.WARN,
        'info': logging.INFO,
        'debug': logging.DEBUG
    }[log_level]

    logging.basicConfig(
        filename=log_file,
        filemode='w',
        format='[%(levelname)s][%(asctime)s.%(msecs)03d] '
               '[%(filename)s:%(lineno)d]: %(message)s',
        level=log_level,
        datefmt='(%Y-%m-%d) %H:%M:%S'
    )
    logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)


def fix_randomness(seed):
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    transformers.set_seed(seed)


def determine_device(config):
    device = "cpu"
    if config.device is not None:
        _device = config.device
        if "cuda" in _device and torch.cuda.is_available():
            device = _device
    return device


def truncate_text_data(config, tokenizer, ground_truth_data):
    result = {"text": [], "token": []}
    max_num_tokens = config.datasource.max_num_tokens
    for idx, token in enumerate(ground_truth_data["token"]):
        text = ground_truth_data["text"][idx]
        if len(token) > max_num_tokens:
            token = token[:max_num_tokens]
            text = tokenizer.decode(token)
        result["text"].append(text)
        result["token"].append(token)
    return result


def prepare_data(config):
    download_config = DownloadConfig(
        resume_download=True,
        max_retries=100
    )
    dataset = load_dataset(
        *config.datasource.full_name, download_config=download_config
    )[config.datasource.partition]

    if config.task == "text-generation":
        # filter our empty sequence
        if config.datasource.main_column is not None:
            dataset = dataset \
                .filter(lambda example: len(example[config.datasource.main_column]) > 0)

    dataset = dataset.shuffle(seed=config.datasource.shuffle_seed)
    assert config.datasource.num_points == 1  # TODO: supporting multiple-point attacks
    select_range = range(config.datasource.num_points)
    dataset = dataset.select(select_range)

    if config.task == "text-generation":
        tokenizer = get_tokenizer(config)

        if config.debug is not None and config.debug.fix_input is True:
            ground_truth_text = [" The Tower Building of the Little Rock Arsenal, also known as U.S."]
            dataset_dict = {
                config.datasource.main_column: ground_truth_text
            }
            dataset = datasets.Dataset.from_dict(dataset_dict)
        else:
            ground_truth_text = dataset[config.datasource.main_column]

        ground_truth_data = {
            'text': ground_truth_text,
            'token': [tokenizer(e)['input_ids'] for e in ground_truth_text]
        }
        logging.info(f'Ground truth text: {ground_truth_data["text"]}')
    elif config.task == "text-classification":
        tokenizer = get_tokenizer(config)

        if config.debug is not None and config.debug.fix_input is True:
            ground_truth_text = ["The Tower Building of the Little Rock Arsenal, also known as U.S."]
            ground_truth_label = [0]  # suppose label is 0 over 0 and 1
            dataset_dict = {
                config.datasource.main_column: ground_truth_text,
                config.datasource.label_column: ground_truth_label
            }
            if config.datasource.dataset == "cola":
                dataset_dict.update({"idx": [0]})
            dataset = datasets.Dataset.from_dict(dataset_dict)
        else:
            ground_truth_text = dataset[config.datasource.main_column][0]
            ground_truth_label = dataset[config.datasource.label_column][0]

        ground_truth_data = {
            'text': ground_truth_text,
            'token': tokenizer(ground_truth_text)['input_ids'],
            'label': ground_truth_label
        }
        logging.info(f'Ground truth text: {ground_truth_data["text"]}')
        logging.info(f"Ground truth label: {ground_truth_data['label']}")
    else:
        raise NotImplementedError

    ground_truth_length = len(ground_truth_data['token'][0])
    logging.info(f"Ground truth text length in tokens: {ground_truth_length}")

    auxiliary = (tokenizer, ground_truth_length)
    return dataset, ground_truth_data, auxiliary


def get_base_model(config):
    if config.model.name == "gpt2":
        if (config.debug is not None
                and config.debug.specific_model_config is True):
            specific_model_config = {
                'activation_function': 'relu',  # default should be gelu_new
                'resid_pdrop': 0.0,  # default should be 0.1
                'embd_pdrop': 0.0,  # default should be 0.1
                'attn_pdrop': 0.0  # default should be 0.1
            }
        else:
            specific_model_config = {}
    else:
        raise NotImplementedError

    if config.task == "text-generation":
        model_config = AutoConfig.from_pretrained(
            config.model.name,
            **specific_model_config
        )
        model = AutoModelForCausalLM.from_pretrained(
            config.model.name,
            config=model_config
        )
    elif config.task == "text-classification":
        model_config = AutoConfig.from_pretrained(
            config.model.name,
            num_labels=config.datasource.num_classes,
            **specific_model_config
        )
        model = AutoModelForSequenceClassification.from_pretrained(
            config.model.name,
            config=model_config
        )
    else:
        raise NotImplementedError

    if not config.model.from_pretrained:
        model.apply(model._init_weights)
        # randomly initialize while keeping the base model information
        # (avoiding push_to_hub error)
    return model


def get_model(config, auxiliary, for_train=False):
    model = get_base_model(config)
    if config.model.has_finetuned:
        pass  # TODO: load the finetuned model

    if for_train:
        model.gradient_checkpointing_enable()  # reduce number of stored activations
        model.enable_input_require_grads()
        model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
    else:
        model.config.use_cache = True

    if config.model.name == "gpt2":
        tokenizer = auxiliary[0]
        model.config.pad_token_id = tokenizer.eos_token_id  # specific to gpt2

    return model


def get_tokenizer(config):
    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer.name, use_fast=False)
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def get_data_collator(config, tokenizer):
    if config.task == "text-generation":
        data_collator = transformers.DataCollatorForLanguageModeling(
            tokenizer, mlm=False)
    elif config.task == "text-classification":
        data_collator = transformers.DataCollatorWithPadding(tokenizer)
    else:
        raise NotImplementedError

    return data_collator


def preprocessing_data(sample, tokenizer=None, main_column=None):
    result = tokenizer(sample[main_column], truncation=True)
    return result


def get_preprocessed_data(raw_dataset, config, tokenizer):
    raw_dataset.cleanup_cache_files()
    # https://huggingface.co/docs/datasets/v1.10.0/processing.html
    # A subsequent call to any of the methods like datasets.Dataset.sort(), datasets.Dataset.map(), etc,
    # will thus reuse the cached file instead of recomputing the operation (even in another python session).
    # this is currently not desirable for debugging

    preprocessed_dataset = raw_dataset.map(
        preprocessing_data,
        fn_kwargs={
            "tokenizer": tokenizer,
            "main_column": config.datasource.main_column
        }
    ).remove_columns(config.datasource.unused_columns_to_remove)
    # otherwise padding during loading will complain

    return preprocessed_dataset


def get_optimizer_and_scheduler(config, variables_to_optimize):
    # Step 1: get optimizer
    if config.optimizer.name in ["bert-adam", "adamw"]:
        optimizer = torch.optim.AdamW(
            variables_to_optimize,
            **config.optimizer.args
        )
    elif config.optimizer.name == "adam":
        optimizer = torch.optim.Adam(
            variables_to_optimize,
            **config.optimizer.args
        )
    else:
        raise NotImplementedError

    # Step 2: get scheduler
    if config.scheduler is not None:
        if config.scheduler.name == "linear":  # specific to "breaching"'s implementation
            def lr_lambda(current_step: int):
                return max(0.0, float(config.max_iterations - current_step)
                           / float(max(1, config.max_iterations)))
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

            scheduler = GradualWarmupScheduler(
                optimizer,
                multiplier=1.0,
                total_epoch=config.scheduler.args.warmup,
                after_scheduler=scheduler
            )
        elif config.scheduler.name == "steplr":
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                **config.scheduler.args
            )
        else:
            raise NotImplementedError
    else:
        scheduler = None

    return optimizer, scheduler


def get_model_forward(config):
    if config.model.name == "gpt2":

        if config.task == "text-generation":
            def forward(model, text_embedding, labels):
                model.zero_grad()
                lm_logits = model(inputs_embeds=text_embedding)["logits"]
                labels = F.softmax(labels, dim=-1)
                shift_logits = lm_logits[:, :-1, :].contiguous()
                shift_labels = labels[:, 1:, :].contiguous().view(-1, labels.shape[-1])
                loss = torch.nn.CrossEntropyLoss()(
                    shift_logits.view(-1, shift_logits.shape[-1]),
                    shift_labels
                )
                return loss, lm_logits
        elif config.task == "text-classification":
            # as the labels are softmaxed ones, we should calculate the loss on our own
            def forward(model, text_embedding, labels):
                model.zero_grad()
                lm_logits = model(inputs_embeds=text_embedding, return_dict=False)[0]
                labels = F.softmax(labels, dim=-1)
                loss = nn.CrossEntropyLoss()(lm_logits, labels)
                return loss, lm_logits.detach()
        else:
            raise NotImplementedError

        return forward
    else:
        raise NotImplementedError


# def get_task_objective(config):
#     if config.model.name == "gpt2":
#         return None  # No use, e.g., gpt2 has its own inside
#     else:
#         raise NotImplementedError


class ValueRecorder:
    def __init__(self):
        self.value = None

    def set(self, value):
        self.value = value

    def get(self):
        return self.value


def normalize_each_row_in_a_torch_matrix(matrix):
    row_sums = torch.sqrt(torch.sum(matrix ** 2, dim=1, keepdim=True))
    normalized_matrix = matrix / row_sums
    return normalized_matrix


def reverse_text_embeddings(config, text_embeddings, model, to_numpy=True):
    if config.model.name == "gpt2":
        if (config.debug is not None
                and config.debug.reverse_text_embeddings == "euclidean"):
            # Euclidean distance
            corresponding_tokens = []
            for seq in text_embeddings:
                seq_token = []
                for embd in seq:
                    embd = embd.unsqueeze(0)
                    dist = torch.norm(model.transformer.wte.weight.data - embd, dim=1)
                    # the diff is [vocab_size, d_model] - [seq_len, d_model]
                    seq_token.append(torch.argmin(dist).item())  # find the nearest one
                corresponding_tokens.append(seq_token)
            corresponding_tokens = np.array(corresponding_tokens)

            if not to_numpy:
                raise NotImplementedError
        else:  # Cosine similarity
            true_embedding = model.transformer.wte.weight.data
            true_embedding = (true_embedding
                              - true_embedding.mean(dim=-1, keepdim=True))
            true_embedding = normalize_each_row_in_a_torch_matrix(
                matrix=true_embedding
            )

            reconstructed_embedding = text_embeddings
            # reconstructed_embedding = (reconstructed_embedding
            #                            - reconstructed_embedding.mean(dim=-1, keepdim=True))
            reconstructed_embedding = normalize_each_row_in_a_torch_matrix(
                matrix=reconstructed_embedding
            )

            similarity = torch.matmul(true_embedding,
                                      torch.transpose(reconstructed_embedding, 1, 2))
            corresponding_tokens = torch.argmax(similarity, dim=1)
            if to_numpy:
                corresponding_tokens = corresponding_tokens.detach().cpu().numpy()
    else:
        raise NotImplementedError

    return corresponding_tokens


def label_data_to_label(label_data):
    softmaxed_label = F.softmax(label_data, dim=-1)
    long_label = torch.argmax(softmaxed_label, dim=-1)
    return long_label


def get_mean_len_embd_in_vocab(config, model):
    if config.model.name == "gpt2":
        vocab_embd = model.transformer.wte.weight.data  # [50257, 768]
        mean_len_embd_in_vocab = vocab_embd.norm(p=2,dim=1).mean()  # 3.9585
    else:
        raise NotImplementedError
    return mean_len_embd_in_vocab
