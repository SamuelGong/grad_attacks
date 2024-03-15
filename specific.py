import logging
import numpy as np
from utils import ValueRecorder
from attack_utils import (post_process,
                          get_closure, initialize_dummy_data)


def discrete_optimization_for_lamp(config, dummy_data, copy_model, auxiliary_model,
                                   dummy_label, attack_objective, target_gradient,
                                   model_forward, tokenizer, mean_len_embd_in_vocab):
    # reference: https://github.com/eth-sri/lamp/blob/852d12f614a9f54138f464131fda1757469417ef/attack.py#L36
    logging.info(f"[Lamp] Started sampling.")

    wrapped_dummy_data = (1, dummy_data.clone().detach())
    post_processed_reconstructed_data = post_process(
        config=config,
        raw_data=wrapped_dummy_data,
        model=copy_model,
        tokenizer=tokenizer,
        token_to_numpy=False
    )
    cos_ids = post_processed_reconstructed_data['token']

    max_len = [dummy_data.shape[1]] * dummy_data.shape[0]
    best_dummy_data, best_total_loss, action = None, None, None
    for seq_id in range(dummy_data.shape[0]):
        for sample_idx in range(config.attack.specific_args.discrete_trial):
            # generating candidates
            permutation = np.arange(dummy_data.shape[1])

            # the first one is the original one
            if not sample_idx == 0:
                if sample_idx % 4 == 0:  # swap two tokens
                    i, j = 1 + np.random.randint(max_len[seq_id] - 2), 1 + np.random.randint(max_len[seq_id] - 2)
                    permutation[i], permutation[j] = permutation[j], permutation[i]
                elif sample_idx % 4 == 1:  # move a token to another place
                    i = 1 + np.random.randint(max_len[seq_id] - 2)
                    j = 1 + np.random.randint(max_len[seq_id] - 1)
                    if i < j:
                        permutation = np.concatenate([permutation[:i], permutation[i + 1:j], permutation[i:i + 1], permutation[j:]])
                    else:
                        permutation = np.concatenate([permutation[:j], permutation[i:i + 1], permutation[j:i], permutation[i + 1:]])
                elif sample_idx % 4 == 2:  # move a sequence to another place
                    b = 1 + np.random.randint(max_len[seq_id] - 1)
                    e = 1 + np.random.randint(max_len[seq_id] - 1)
                    if b > e:
                        b, e = e, b
                    p = 1 + np.random.randint(max_len[seq_id] - 1 - (e - b))
                    if p >= b:
                        p += e - b
                    if p < b:
                        permutation = np.concatenate([permutation[:p], permutation[b:e], permutation[p:b], permutation[e:]])
                    elif p >= e:
                        permutation = np.concatenate([permutation[:b], permutation[e:p], permutation[b:e], permutation[p:]])
                    else:
                        assert False
                elif sample_idx % 4 == 3:  # take some prefix and put it at the end
                    i = 1 + np.random.randint(max_len[seq_id] - 2)
                    permutation = np.concatenate([permutation[:1], permutation[i:-1], permutation[1:i], permutation[-1:]])

            new_dummy_tokens = cos_ids.clone()
            new_dummy_tokens[seq_id] = cos_ids[seq_id, permutation]
            new_dummy_data = dummy_data.clone()
            new_dummy_data[seq_id] = new_dummy_data[seq_id, permutation, :]

            # using an LM to select candidates
            new_total_loss = lamp_loss(
                config=config,
                copy_model=copy_model,
                new_dummy_data=new_dummy_data,
                new_dummy_tokens=new_dummy_tokens,
                dummy_label=dummy_label,
                target_gradient=target_gradient,
                auxiliary_model=auxiliary_model,
                attack_objective=attack_objective,
                model_forward=model_forward,
                mean_len_embd_in_vocab=mean_len_embd_in_vocab
            )[2]

            if (best_total_loss is None) or (new_total_loss < best_total_loss):
                best_dummy_data = new_dummy_data
                best_total_loss = new_total_loss
                if not sample_idx == 0:
                    action = sample_idx % 4

            if (sample_idx + 1 == config.attack.specific_args.discrete_trial
                    or config.attack.specific_args.discrete_trial_print_interval is not None
                    and sample_idx % config.attack.specific_args.discrete_trial_print_interval == 0):
                logging.info(
                    f"\t| Seq: {seq_id + 1} | Sample: {sample_idx + 1} | "
                    f"Best Total Loss: {best_total_loss.item():2.5f} |"
                    # f"Atk. loss: {attack_loss.item():2.5f} | "
                    # f"Task loss: {task_loss_recorder.get():2.5f} | "
                    # f"T: {timestamp - opt_start_time:4.2f}s |"
                )

        if not (action is None):
            changed = ['Swapped tokens', 'Moved token', 'Moved sequence', 'Put prefix at the end'][action]
            logging.info(f"[Lamp] Sampling for Sequence {seq_id + 1} done. "
                         f"The best one is generated by '{changed}'")
        else:
            logging.info(f"[Lamp] Sampling for Sequence {seq_id + 1} done. "
                         f"The best one is the original one")

    best_dummy_data = best_dummy_data.detach()
    best_dummy_data.requires_grad = True
    return best_dummy_data


def lamp_loss(config, copy_model, new_dummy_data, new_dummy_tokens, dummy_label,
              target_gradient, auxiliary_model, attack_objective, model_forward,
              mean_len_embd_in_vocab):
    if config.attack.specific_args.auxiliary_model == "gpt2":
        loss = auxiliary_model(
            input_ids=new_dummy_tokens,
            labels=new_dummy_tokens,
            return_dict=False
        )[0]
        # according to the original code, they are not using the actual perplexity
        # they are using the original loss instead
        perplexity = loss

        virtual_task_loss_recorder = ValueRecorder()  # placeholder, not actual use
        aux = (mean_len_embd_in_vocab,)
        closure = get_closure(
            config=config,
            attack_objective=attack_objective,
            model=copy_model,
            model_forward=model_forward,
            dummy_data=new_dummy_data,
            dummy_label=dummy_label,
            target_gradient=target_gradient,
            task_loss_recorder=virtual_task_loss_recorder,
            aux=aux
        )
        attack_loss = closure()
        total_loss = (attack_loss
                      + config.attack.specific_args.perplexity_scale * perplexity)
        return perplexity, attack_loss, total_loss
    else:
        raise NotImplementedError


def lamp_initialize_dummy(config, data_size, label_size, copy_model,
                          target_gradient, attack_objective, model_forward,
                          mean_len_embd_in_vocab, device):
    logging.info(f"[Lamp] Start initial guess.")
    best_dummy_data, best_dummy_label, best_attack_loss = None, None, None
    assert config.attack.specific_args.init_seed is None
    # otherwise, different initial guess does not make sense

    for iter in range(config.attack.specific_args.num_init_guess):
        dummy_data = initialize_dummy_data(config, data_size, device, log_seed=False)
        dummy_label = initialize_dummy_data(config, label_size, device, log_seed=False)

        virtual_task_loss_recorder = ValueRecorder()  # placeholder, not actual use
        aux = (mean_len_embd_in_vocab,)
        closure = get_closure(
            config=config,
            attack_objective=attack_objective,
            model=copy_model,
            model_forward=model_forward,
            dummy_data=dummy_data,
            dummy_label=dummy_label,
            target_gradient=target_gradient,
            task_loss_recorder=virtual_task_loss_recorder,
            aux=aux
        )
        attack_loss = closure()

        if best_attack_loss is not None:
            if attack_loss < best_attack_loss:
                best_attack_loss = attack_loss
                best_dummy_label = dummy_label
                best_dummy_data = dummy_data
        else:
            best_attack_loss = attack_loss
            best_dummy_label = dummy_label
            best_dummy_data = dummy_data

        if (iter + 1 == config.attack.specific_args.num_init_guess
                or config.attack.specific_args.init_print_interval is not None
                and iter % config.attack.specific_args.init_print_interval == 0):
            logging.info(
                f"\t| Iter: {iter + 1} | "
                f"Best Attack Loss: {best_attack_loss.item():2.5f} |"
            )

    logging.info(f"[Lamp] Initial guess ended.")
    return best_dummy_data, best_dummy_label
