import copy
import os
import time
import torch
import logging
import numpy as np
from eval import eval_attack
from utils import (
    determine_device, get_optimizer_and_scheduler,
    get_model_forward, reverse_text_embeddings, ValueRecorder,
    label_data_to_label
)


def initialize_dummy_data(config, size, device):
    init_type = config.attack.specific_args.init_type
    if init_type == 'randn-trunc':

        rng_state = torch.get_rng_state()
        seed = config.attack.specific_args.init_seed
        if seed is None:
            seed = int.from_bytes(os.urandom(8), 'big')
            logging.info(f"Randomly generated seed: {seed}")

        torch.manual_seed(seed)

        result = (torch.randn(size=size, dtype=torch.float32,
                              device=device)
                  * 0.1).clamp(-0.1, 0.1)

        torch.set_rng_state(rng_state)
        result.requires_grad = True
    else:
        raise NotImplementedError

    # reset the irrelevant gradient
    result.grad = torch.zeros_like(result)  # TODO: to see if it is removable
    return result


def post_process(config, recovered_data, model, tokenizer):
    result = {}
    if config.model.name == "gpt2":
        iter = recovered_data[0]
        embeddings_list = recovered_data[1]
        token_list = reverse_text_embeddings(
            config=config,
            text_embeddings=embeddings_list,
            model=model
        )
        text_list = [tokenizer.decode(e) for e in token_list]
        result.update({
            'iter': iter,
            'token': token_list,
            'text': text_list,
        })

        if config.task == "text-classification":
            label_data = recovered_data[2]
            label = label_data_to_label(label_data).item()
            result['label'] = label
    else:
        raise NotImplementedError

    return result


def get_closure(config, attack_objective, optimizer,
                model, model_forward, dummy_data, dummy_label,
                target_gradient, task_loss_recorder):
    def closure():
        optimizer.zero_grad()

        if config.model.name == "gpt2":
            dummy_loss, dummy_pred = model_forward(
                model, dummy_data, dummy_label
            )
        else:
            raise NotImplementedError
        task_loss_recorder.set(dummy_loss.item())

        dummy_gradient = torch.autograd.grad(
            outputs=dummy_loss,
            inputs=model.parameters(),
            create_graph=True,
            allow_unused=True,
            materialize_grads=True
        )
        # create_graph: necessary
        # allows_unused: some layers are indeed unused (e.g., wte of gpt2 during classification)
        # materialize_grads: when this is unavailable, currently some layers can be None

        attack_loss = attack_objective(target_gradient, dummy_gradient)
        attack_loss.backward(
            inputs=[dummy_data, dummy_label],
            create_graph=False
        )

        with torch.no_grad():
            if config.attack.specific_args.grad_clip is not None:
                grad_clip = config.attack.specific_args.grad_clip
                for element in [dummy_data, dummy_label]:
                    grad_norm = element.grad.norm()
                    if grad_norm > grad_clip:
                        element.grad.mul_(grad_clip / (grad_norm + 1e-6))

        return attack_loss

    return closure


def get_attack_objective(config):
    if config.attack.name == "tag":
        def objective(target_gradient, dummy_gradient):
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
        def objective(target_gradient, dummy_gradient):
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
    else:
        raise NotImplementedError

    return objective


def gradient_attack(config, model, gradient, auxiliary, ground_truth_data):
    if config.task in ["text-generation", "text-classification"]:
        tokenizer, ground_truth_length = auxiliary
    else:
        raise NotImplementedError

    device = determine_device(config=config)
    copy_model = copy.deepcopy(model)

    if (config.attack.name in ["tag", "april"]
            and config.task in ["text-generation", "text-classification"]):
        data_size = (1, ground_truth_length, config.attack.specific_args.d_model)
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
            label_size = (config.datasource.num_classes,)
            dummy_label = initialize_dummy_data(config, label_size, device)

            # # only for debug use
            # dummy_label = torch.tensor([1.0, 0.0]).to(device)  # suppose label is 0 over 0 and 1
            # dummy_label.requires_grad = True

            variables_to_optimize.append(dummy_label)
        else:  # "text-generation"
            label_size = (1, ground_truth_length, tokenizer.vocab_size)
            dummy_label = initialize_dummy_data(config, label_size, device)
            variables_to_optimize.append(dummy_label)

        optimizer, scheduler = get_optimizer_and_scheduler(
            config=config.attack.specific_args,
            variables_to_optimize=variables_to_optimize,
        )
        model_forward = get_model_forward(config)
        attack_objective = get_attack_objective(config)

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
                target_gradient=gradient,
                task_loss_recorder=task_loss_recorder
            )
            attack_loss = optimizer.step(closure=closure)
            if scheduler is not None:
                scheduler.step()

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
                        recovered_data=recovered_data,
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
