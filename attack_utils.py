# this file is created to avoid circular imports
import os
import torch
import logging
from utils import reverse_text_embeddings, label_data_to_label


def post_process(config, raw_data, model, tokenizer, token_to_numpy=True):
    result = {}
    if config.model.name == "gpt2":
        iter = raw_data[0]
        embeddings_list = raw_data[1]
        token_list = reverse_text_embeddings(
            config=config,
            text_embeddings=embeddings_list,
            model=model,
            to_numpy=token_to_numpy
        )
        text_list = [tokenizer.decode(e) for e in token_list]
        result.update({
            'iter': iter,
            'token': token_list,
            'text': text_list,
        })

        if config.task == "text-classification":
            label_data = raw_data[2]
            label = label_data_to_label(label_data).item()
            result['label'] = label
    else:
        raise NotImplementedError

    return result


def get_closure(config, attack_objective, model, model_forward, dummy_data, dummy_label,
                target_gradient, task_loss_recorder, aux, optimizer=None,):
    def closure():
        if optimizer is not None:
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

        if config.attack.name == "lamp":
            mean_len_embd_in_vocab = aux[0]
            auxiliary = (dummy_data, mean_len_embd_in_vocab)
        else:
            auxiliary = None
        attack_loss = attack_objective(target_gradient, dummy_gradient, auxiliary)
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


def initialize_dummy_data(config, size, device, log_seed=True):
    init_type = config.attack.specific_args.init_type
    if init_type == 'randn-trunc':

        rng_state = torch.get_rng_state()
        seed = config.attack.specific_args.init_seed
        if seed is None:
            seed = int.from_bytes(os.urandom(8), 'big')
            if log_seed:
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
