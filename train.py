import copy
import os
import torch
import pickle
import logging
from utils import (
    get_model, get_data_collator,
    get_preprocessed_data, determine_device
)
from torch.utils.data import DataLoader


def get_params_list_for_grad(model):
    result = []
    for name, e in model.named_parameters():
        if not e.requires_grad:
            e.requires_grad = True
        result.append(e)
    return result


def get_gradient(config, dataset, auxiliary):
    model = get_model(config=config, auxiliary=auxiliary, for_train=False)
    device = determine_device(config=config)
    model = model.to(device).eval()

    if config.task in ["text-generation", "text-classification"]:
        tokenizer = auxiliary[0]
        data_collator = get_data_collator(config=config, tokenizer=tokenizer)
        preprocessed_dataset = get_preprocessed_data(
            raw_dataset=dataset,
            config=config,
            tokenizer=tokenizer
        )
    else:
        raise NotImplementedError

    dataloader = DataLoader(
        preprocessed_dataset,
        batch_size=config.train.batch_size,
        shuffle=False,
        collate_fn=data_collator
    )

    params_list_for_grad = get_params_list_for_grad(model)
    copy_model = copy.deepcopy(model)
    for batch_id, batch in enumerate(dataloader):  # currently should only have one batch
        batch = {k: v.type(torch.long).to(device) for k, v in batch.items()}
        if config.model.name == "gpt2":
            text_embeddings = copy_model.transformer.wte(batch['input_ids'])
            if config.task in ["text-generation", "text-classification"]:
                outputs = model(
                    inputs_embeds=text_embeddings,
                    attention_mask=batch['attention_mask'],
                    labels=batch['labels']
                )
            else:
                raise NotImplementedError
            # torch.save(text_embeddings, "true_input.pt")
            # torch.save(batch['labels'], "true_label.pt")
            loss, logits = outputs[:2]
        else:
            # outputs = model(**batch)
            raise NotImplementedError

        gradient = torch.autograd.grad(
            outputs=loss,
            inputs=params_list_for_grad,
            allow_unused=True,
            materialize_grads=True  # can be used in new torch
        )
        # allows_unused: some layers are indeed unused (e.g., wte of gpt2 during classification)
        # materialize_grads: if this is unavailable, currently some layers can be None

    return gradient, model
