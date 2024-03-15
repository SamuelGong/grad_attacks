import copy
import os
import time
import torch
import logging
import numpy as np
from eval import eval_attack
from attack_utils import (post_process,
                          get_closure, initialize_dummy_data)
from utils import (
    determine_device, get_optimizer_and_scheduler,
    get_model_forward, ValueRecorder,
    get_mean_len_embd_in_vocab
)
from specific import (discrete_optimization_for_lamp,
                      lamp_initialize_dummy)


def get_attack_objective(config):
    if config.attack.name == "tag":
        def objective(target_gradient, dummy_gradient, auxiliary=None):
            # target_gradient/dummy_gradient:
            # gradient of each layer (for gpt2, there are 148 layers)
            weights = torch.arange(len(dummy_gradient), 0, -1) / len(dummy_gradient)
            if config.model.name == "gpt2" \
                    and target_gradient[0].shape == torch.Size([50257, 768]) \
                    and not (config.debug is not None and config.debug.wrong_alpha is True):
                # if the first layer is transformer.wte.weight, it should
                # have the smallest weight as it is actually the furthest from the input
                temp = weights[-1].clone()
                weights[1:] = weights[:-1].clone()
                weights[0] = temp
            weights = weights.to(dummy_gradient[0].device)

            objective_value = 0
            scale = config.attack.specific_args.scale

            for dum, tar, weight in zip(dummy_gradient, target_gradient, weights):
                objective_value += ((dum - tar).pow(2).sum()
                                    + scale * weight * (dum - tar).abs().sum())
            objective_value *= 0.5
            return objective_value
    elif config.attack.name == "april":
        def objective(target_gradient, dummy_gradient, auxiliary=None):
            # part one
            objective_value = 0
            for dum, tar in zip(dummy_gradient, target_gradient):
                objective_value += (dum - tar).pow(2).sum()

            # part two
            scale = config.attack.specific_args.scale
            if config.model.name == "gpt2":
                # layer 1 is the positional embedding
                objective_value += scale * (dummy_gradient[1]
                                            - target_gradient[1]).pow(2).sum()
            else:
                raise NotImplementedError

            objective_value *= 0.5
            return objective_value
    elif config.attack.name == "lamp":
        # variant one

        def objective(target_gradient, dummy_gradient, auxiliary):

            # part one: reconstruction loss
            if config.attack.specific_args.variant == "cos":
                reconstruction_loss = 0
                num_layers = 0
                for dum, tar in zip(dummy_gradient, target_gradient):
                    reconstruction_loss += (1.0 - (dum * tar).sum()
                                        / (dum.view(-1).norm(p=2) * tar.view(-1).norm(p=2)))
                    num_layers += 1
                reconstruction_loss /= num_layers
            else:  # TODO: "tag"
                raise NotImplementedError

            # part two: embedding regularization loss
            dummy_data, mean_len_embd_in_vocab = auxiliary[:2]
            mean_len_embd_in_vocab = mean_len_embd_in_vocab.to(dummy_data.device)
            regularization_loss = ( dummy_data.norm(p=2, dim=2).mean()
                                    - mean_len_embd_in_vocab).square()

            objective_value = (reconstruction_loss + config.attack.specific_args.reg_scale
                               * regularization_loss)
            return objective_value
    else:
        raise NotImplementedError

    return objective


def gradient_attack(config, model, target_gradient, auxiliary, ground_truth_data):
    if config.task in ["text-generation", "text-classification"]:
        tokenizer, ground_truth_length = auxiliary
    else:
        raise NotImplementedError

    device = determine_device(config=config)
    copy_model = copy.deepcopy(model)

    if (config.attack.name in ["tag", "april", "lamp"]
            and config.task in ["text-generation", "text-classification"]):
        if config.attack.name == 'lamp':
            mean_len_embd_in_vocab = get_mean_len_embd_in_vocab(config, copy_model)
            aux = (mean_len_embd_in_vocab,)
        else:
            aux = None

        model_forward = get_model_forward(config)
        attack_objective = get_attack_objective(config)

        data_size = (1, ground_truth_length, config.attack.specific_args.d_model)
        if config.task == "text-classification":
            label_size = (config.datasource.num_classes,)
        else:  # text-generation
            label_size = (1, ground_truth_length, tokenizer.vocab_size)

        if (config.attack.name == "lamp"
                and config.attack.specific_args.num_init_guess > 1):
            dummy_data, dummy_label = lamp_initialize_dummy(
                config=config,
                data_size=data_size,
                label_size=label_size,
                copy_model=copy_model,
                target_gradient=target_gradient,
                attack_objective=attack_objective,
                model_forward=model_forward,
                mean_len_embd_in_vocab=mean_len_embd_in_vocab,
                device=device
            )
            variables_to_optimize = [dummy_data, dummy_label]
        else:
            dummy_data = initialize_dummy_data(config, data_size, device)
            variables_to_optimize = [dummy_data]

            # # only for debug use
            # truth = "The Tower Building of the Little Rock Arsenal, also known as U.S."
            # tokens = tokenizer.encode(truth)
            # tokens = torch.tensor(tokens).to(device)
            # dummy_data = copy_model.transformer.wte(tokens).detach()
            # dummy_data.requires_grad = True

            if config.task == "text-classification":
                dummy_data = dummy_data.unsqueeze(0)
                dummy_label = initialize_dummy_data(config, label_size, device)

                # # only for debug use
                # dummy_label = torch.tensor([1.0, 0.0]).to(device)  # suppose label is 0 over 0 and 1
                # dummy_label.requires_grad = True
                variables_to_optimize.append(dummy_label)
            else:  # "text-generation"
                dummy_label = initialize_dummy_data(config, label_size, device)
                variables_to_optimize.append(dummy_label)

        optimizer, scheduler = get_optimizer_and_scheduler(
            config=config.attack.specific_args,
            variables_to_optimize=variables_to_optimize,
        )
        opt_start_time = time.perf_counter()
        task_loss_recorder = ValueRecorder()
        for iter in range(config.attack.specific_args.max_iterations):
            if config.model.name == "gpt2":
                if config.task == "text-generation":
                    dummy_label = dummy_label

                    # only for debug use
                    # truth = "The Tower Building of the Little Rock Arsenal, also known as U.S."
                    # tokens = tokenizer.encode(truth)
                    # _dummy_label = torch.tensor(tokens).to(device)
                elif config.task == "text-classification":
                    # print(dummy_label)
                    dummy_label = dummy_label
                    # _dummy_label = label_data_to_label(dummy_label)
                    dummy_label = dummy_label.unsqueeze(0)  # required for one-data recovery
                else:
                    raise NotImplementedError

                # true_data = torch.load(f"true_input.pt")
                # true_label = torch.load(f"true_label.pt")

            closure = get_closure(
                config=config,
                attack_objective=attack_objective,
                optimizer=optimizer,
                model=model,
                model_forward=model_forward,
                dummy_data=dummy_data,
                dummy_label=dummy_label,
                target_gradient=target_gradient,
                task_loss_recorder=task_loss_recorder,
                aux=aux
            )
            attack_loss = optimizer.step(closure=closure)
            if scheduler is not None:
                scheduler.step()

            if (config.attack.name == "lamp"
                    and (iter + 1) % config.attack.specific_args.continuous_period == 0):
                if config.attack.specific_args.auxiliary_model == "gpt2":
                    auxiliary_model = copy_model  # shortcut. leaving other possibilities TODO
                else:
                    raise NotImplementedError
                dummy_data = discrete_optimization_for_lamp(
                    config=config,
                    dummy_data=dummy_data,
                    copy_model=copy_model,
                    tokenizer=tokenizer,
                    auxiliary_model=auxiliary_model,
                    dummy_label=dummy_label,
                    attack_objective=attack_objective,
                    target_gradient=target_gradient,
                    model_forward=model_forward,
                    mean_len_embd_in_vocab=mean_len_embd_in_vocab
                )

            # Text logging
            if (iter + 1 == config.attack.specific_args.max_iterations
                    or config.attack.specific_args.print_interval is not None
                    and iter % config.attack.specific_args.print_interval == 0):
                timestamp = time.perf_counter()
                logging.info(
                    f"| It: {iter + 1} | "
                    f"Atk. loss: {attack_loss.item():2.5f} | "
                    f"Task loss: {task_loss_recorder.get():2.5f} | "
                    f"T: {timestamp - opt_start_time:4.2f}s |"
                )

            # Snapshots capturing
            if (iter + 1 == config.attack.specific_args.max_iterations
                    or config.attack.specific_args.snapshot_interval is not None
                    and iter % config.attack.specific_args.snapshot_interval == 0):
                logging.info(f"Snapshots captured for It {iter + 1}.")

                if config.task in ["text-generation", "text-classification"]:
                    if config.task == "text-generation":
                        recovered_data = (iter + 1, dummy_data.clone().detach())
                    else:
                        recovered_data = (iter + 1, dummy_data.clone().squeeze().detach(),
                                          dummy_label.detach())

                    post_processed_reconstructed_data = post_process(
                        config=config,
                        raw_data=recovered_data,
                        model=copy_model,
                        tokenizer=tokenizer
                    )
                    eval_attack(
                        config=config,
                        ground_truth_data=ground_truth_data,
                        reconstructed_data=post_processed_reconstructed_data,
                    )
                else:
                    raise NotImplementedError
    else:
        raise NotImplementedError

    # currently only return the last recovered data for possible use
    return post_processed_reconstructed_data
