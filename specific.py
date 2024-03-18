import time
import copy
import torch
import logging
import numpy as np
from eval import eval_attack
from itertools import permutations
from train import get_params_list_for_grad
from utils import ValueRecorder, determine_device
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


def film_analyze_gradient(config, gradient, tokenizer):
    logging.info(f"[Film] Start generating bag of words.")

    if config.model.name == "gpt2":
        bag_of_words_threshold = config.attack.specific_args.bag_of_words_threshold
        text_embd_grad = gradient[0]  # [50257, 768]
        row_sum = torch.sum(torch.abs(text_embd_grad), dim=1).squeeze()
        bag_of_words = torch.where(row_sum > bag_of_words_threshold)[0].detach().cpu().numpy()
        bag_of_words_in_tokens = [tokenizer.decode(e) for e in bag_of_words]

        pos_embd_grad = gradient[1]  # [1024, 768]
        max_seq_len_in_batch = None
        for row_idx, embd in enumerate(pos_embd_grad):
            if torch.all(embd == 0):
                max_seq_len_in_batch = row_idx + 1
                break
        if max_seq_len_in_batch is None:
            max_seq_len_in_batch = 1024
    else:
        raise NotImplementedError

    bag_of_words_print_str = "["
    cnt = 0
    l = len(bag_of_words)
    for token_id, token in zip(bag_of_words, bag_of_words_in_tokens):
        bag_of_words_print_str += f"({token_id}, {token})"
        if cnt < l - 1:
            bag_of_words_print_str += ", "
        cnt += 1
    bag_of_words_print_str += "]"
    logging.info(f"[Film] Gradient analyzed: "
                 f"bag of {len(bag_of_words)} words={bag_of_words_print_str}, "
                 f"maximum sequence length in the batch: {max_seq_len_in_batch}")
    return bag_of_words, max_seq_len_in_batch


def film_reserve_top_k_sequences(config, new_candidate_sequences, model):
    total_score_list = []  # the larger the better
    perplexity_score_list, diversity_score_list = [], []
    for idx, candidate_sequence in enumerate(new_candidate_sequences):
        model_input = torch.tensor(candidate_sequence).to(model.device).unsqueeze(0)
        if config.model.name == "gpt2":
            loss = model(
                input_ids=model_input,
                labels=model_input,
                return_dict=False
            )[0]
            perplexity_score = -loss.item()  # the larger the better

            n_grams_map = {}  # should be something like {(1, 2): 2, (2, 3): 2, (3, 4): 1, (4, 1): 1}
            for i in range(len(candidate_sequence)):
                j = i + config.attack.specific_args.ngram_size_penalty
                if j > len(candidate_sequence):
                    break
                n_gram = tuple(candidate_sequence[i:j])
                if n_gram not in n_grams_map:
                    n_grams_map[n_gram] = 1
                else:
                    n_grams_map[n_gram] += 1

            diversity_score = len(n_grams_map)  # the larger the better
            diversity_score *= config.attack.specific_args.ngram_size_penalty_factor

            total_score = perplexity_score + diversity_score
            perplexity_score_list.append(perplexity_score)
            diversity_score_list.append(diversity_score)
            total_score_list.append(total_score)

    sorted_items = sorted(zip(new_candidate_sequences, total_score_list),
                          key=lambda x: x[1], reverse=True)
    top_k_sequences = [item for item, _ in
                       sorted_items[:config.attack.specific_args.num_beams]]
    return top_k_sequences


def film_beam_search(config, model, tokenizer, bag_of_words, max_seq_len_in_batch):
    # Step 1: find starting tokens
    starting_token_ids = []
    bag_of_words_in_tokens = [tokenizer.decode(e) for e in bag_of_words]
    starting_tokens_for_printing = "["
    for token_id, token in zip(bag_of_words, bag_of_words_in_tokens):
        token = token.lstrip()  # eliminating leading space
        if len(token) > 0 and token.lstrip()[0].isupper():
            starting_token_ids.append(token_id)
            if len(starting_tokens_for_printing) > 1:
                starting_tokens_for_printing += ", "
            starting_tokens_for_printing += f"({token_id}, {token})"
    starting_tokens_for_printing += "]"
    logging.info(f"[Film] Found {len(starting_token_ids)} "
                 f"starting tokens: {starting_tokens_for_printing}")

    # Step 2: start the loop
    assert max_seq_len_in_batch > 1
    start_time = time.perf_counter()
    candidate_sequences = [[e] for e in starting_token_ids]
    for iter in range(max_seq_len_in_batch - 1):
        new_candidate_sequences = []
        for candidate_sequence in candidate_sequences:
            for word in bag_of_words:
                new_candidate_sequences.append(candidate_sequence + [word])

        candidate_sequences = film_reserve_top_k_sequences(
            config=config,
            new_candidate_sequences=new_candidate_sequences,
            model=model
        )

        if (iter + 1 == max_seq_len_in_batch - 1
                or config.attack.specific_args.beam_search_print_interval is not None
                and iter % config.attack.specific_args.beam_search_print_interval == 0):
            timestamp = time.perf_counter()
            logging.info(f"\t| Beam size: {iter + 2} | "
                         f"T: {timestamp - start_time:4.2f}s |")

    decoded_candidate_sequences = tokenizer.batch_decode(
        candidate_sequences,
        skip_special_tokens=True
    )
    candidate_sequences_to_print = "["
    l = len(candidate_sequences)
    cnt = 0
    for candidate_sequence, decoded_candidate_sequence in zip(candidate_sequences, decoded_candidate_sequences):
        candidate_sequences_to_print += f"({candidate_sequence}, {decoded_candidate_sequence})"
        if cnt < l - 1:
            candidate_sequences_to_print += ", "
        cnt += 1
    candidate_sequences_to_print += "]"
    logging.info(f"[Film] Generated {len(candidate_sequences)} "
                 f"candidate sequences after beam search, "
                 f"and they are: {candidate_sequences_to_print}")

    return candidate_sequences


film_cand_orders = {
    3: [
        list(ll)
        for ll in list(permutations(range(1, 5)))
        if ll[0] == 1 and ll[-1] == 4 and ll != tuple(range(1, 5))
    ],
    4: [
        list(ll)
        for ll in list(permutations(range(1, 6)))
        if ll[0] == 1 and ll[-1] == 5 and ll != tuple(range(1, 6))
    ],
    5: [
        list(ll)
        for ll in list(permutations(range(1, 7)))
        if ll[0] == 1 and ll[-1] == 6 and ll != tuple(range(1, 7))
    ],
    6: [
        list(ll)
        for ll in list(permutations(range(1, 8)))
        if ll[0] == 1 and ll[-1] == 7 and ll != tuple(range(1, 8))
    ],
    7: [
        list(ll)
        for ll in list(permutations(range(1, 9)))
        if ll[0] == 1 and ll[-1] == 8 and ll != tuple(range(1, 9))
    ]
}  # the value is a list


def film_score(config, sequence, model, target_gradient):
    score = 0.0
    if config.model.name == "gpt2":
        # part one: perplexity
        sequence = torch.tensor(sequence).unsqueeze(0).to(model.device)
        loss = model(
            input_ids=sequence,
            labels=sequence,
            return_dict=False
        )[0]
        perplexity = torch.exp(loss).item()
        score += perplexity

        # part two: gradient norm
        params_list_for_grad = get_params_list_for_grad(model)
        reconstruction_gradient = torch.autograd.grad(
            outputs=loss,
            inputs=params_list_for_grad,
            allow_unused=True,
            materialize_grads=True  # can be used in new torch
        )
        gradient_diff_norm = 0.0  # in the paper is merely grad_norm, which I think does not make sense
        for recon, tar in zip(reconstruction_gradient, target_gradient):
            gradient_diff_norm += torch.norm(tar - recon)
        gradient_diff_norm = gradient_diff_norm.item()
        score += (config.attack.specific_args.gradient_norm_factor
                  * gradient_diff_norm)

        # # for debug only
        # print(perplexity, gradient_diff_norm, score)
        # # e.g.: 10.202592849731445 308.8561096191406 319.05870246887207
    else:
        raise NotImplementedError

    return score


def film_phrase_reordering(config, sequence, model, target_gradient, tokenizer):
    best_sequence = sequence
    best_sequence_in_words = tokenizer.decode(sequence).lstrip().rstrip()
    best_score = film_score(
        config=config,
        sequence=sequence,
        model=model,
        target_gradient=target_gradient
    )  # the lower, the better

    start_time = time.perf_counter()
    logging.info(f"[Film] Start reordering sequence phrase-wise. "
                 f"The original sequence is {sequence}, "
                 f"which in natural language is '{best_sequence_in_words}'")
    for iter in range(config.attack.specific_args.num_phrase_reorder_steps):
        sequence_in_words = copy.deepcopy(best_sequence_in_words)
        kopt = np.random.randint(3, 8)

        words = sequence_in_words.split(" ")
        kopt = min(len(words) + 1, kopt)  # up to len(sequence_in_words) + 1 non-overlapping cuts
        cut_points = sorted(np.random.choice(
            a=np.arange(len(words) + 1),
            size=kopt,
            replace=False
        ))
        word_groups = [words[:cut_points[0]]]
        for idx, cut_point in enumerate(cut_points):
            if idx == len(cut_points) - 1:
                start = cut_point
                end = len(words) + 1
            else:
                start = cut_point
                end = cut_points[idx + 1]
            word_groups.append(words[start:end])

        candidate_orders = film_cand_orders[kopt]
        if config.attack.specific_args.variant == "random":
            # unweighted random sample
            shuffle_order_index = np.random.choice(
                a=len(candidate_orders),
                size=1
            )[0]
        else:
            raise NotImplementedError
        shuffle_order = candidate_orders[shuffle_order_index]

        sequence_in_words = ""
        for idx, original_group in enumerate(shuffle_order):
            new_segment = " ".join(word_groups[original_group - 1])
            sequence_in_words += new_segment
            if not idx == len(shuffle_order) - 1 and len(new_segment) > 0:
                sequence_in_words += " "
        sequence = tokenizer.encode(sequence_in_words)
        score = film_score(
            config=config,
            sequence=sequence,
            model=model,
            target_gradient=target_gradient
        )
        if score < best_score:
            best_score = score
            best_sequence = sequence
            best_sequence_in_words = sequence_in_words

        if (iter + 1 == config.attack.specific_args.num_phrase_reorder_steps
                or config.attack.specific_args.phrase_reorder_print_interval is not None
                and iter % config.attack.specific_args.phrase_reorder_print_interval == 0):
            timestamp = time.perf_counter()
            logging.info(
                f"\t| It: {iter + 1} | "
                f"Best Score: {best_score} | "
                f"T: {timestamp - start_time:4.2f}s |"
            )

    logging.info(f"[Film] After reordering sequence phrase-wise, "
                 f"the best sequence is {best_sequence}, "
                 f"which in natural language is '{best_sequence_in_words}'")
    return best_sequence


def film_token_reordering(config, sequence, model, target_gradient, tokenizer, bag_of_words):
    best_sequence = sequence
    best_sequence_in_words = tokenizer.decode(sequence).lstrip().rstrip()
    best_score = film_score(
        config=config,
        sequence=sequence,
        model=model,
        target_gradient=target_gradient
    )  # the lower, the better

    start_time = time.perf_counter()
    logging.info(f"[Film] Start reordering sequence token-wise. "
                 f"The original sequence is {sequence}, "
                 f"which in natural language is '{best_sequence_in_words}'")
    for iter in range(config.attack.specific_args.num_token_reorder_steps):
        sequence = copy.deepcopy(best_sequence)
        action = np.random.randint(0, 3)  # 0: swap, 1: insert, 2: delete
        if action == 0:
            i, j = np.random.choice(a=len(sequence), size=2, replace=False)
            temp = sequence[i]
            sequence[i] = sequence[j]
            sequence[j] = temp
        elif action == 1:
            i = np.random.choice(a=len(sequence) + 1, size=1)[0]
            random_token = np.random.choice(a=bag_of_words, size=1)[0]
            sequence.insert(i, random_token)
        else:
            i = np.random.choice(a=len(sequence), size=1)[0]
            sequence.pop(i)

        score = film_score(
            config=config,
            sequence=sequence,
            model=model,
            target_gradient=target_gradient
        )
        if score < best_score:
            best_score = score
            best_sequence = sequence
            sequence_in_words = tokenizer.decode(sequence).lstrip().rstrip()
            best_sequence_in_words = sequence_in_words

        if (iter + 1 == config.attack.specific_args.num_token_reorder_steps
                or config.attack.specific_args.token_reorder_print_interval is not None
                and iter % config.attack.specific_args.token_reorder_print_interval == 0):
            timestamp = time.perf_counter()
            logging.info(
                f"\t| It: {iter + 1} | "
                f"Best Score: {best_score} | "
                f"T: {timestamp - start_time:4.2f}s |"
            )

    logging.info(f"[Film] After reordering sequence token-wise, "
                 f"the best sequence is {best_sequence}, "
                 f"which in natural language is '{best_sequence_in_words}'")
    return best_sequence


def film_reordering(config, sequence, model, tokenizer,
                    target_gradient, bag_of_words):
    # Step 1: phrase-wise reordering
    sequence = film_phrase_reordering(
        config=config,
        sequence=sequence,
        model=model,
        target_gradient=target_gradient,
        tokenizer=tokenizer
    )

    # Step 2: token-wise reordering
    sequence = film_token_reordering(
        config=config,
        sequence=sequence,
        model=model,
        target_gradient=target_gradient,
        tokenizer=tokenizer,
        bag_of_words=bag_of_words
    )
    return sequence


def film_attack(config, model, auxiliary, target_gradient, ground_truth_data):
    device = determine_device(config=config)
    model = model.to(device)
    tokenizer = auxiliary[0]

    # Step 1: analyze the gradient
    bag_of_words, max_seq_len_in_batch = film_analyze_gradient(
        config=config,
        gradient=target_gradient,
        tokenizer=tokenizer,  # for debug only
    )

    # Step 2: conduct beam search
    top_k_sequences = film_beam_search(
        config=config,
        model=model,
        tokenizer=tokenizer,
        bag_of_words=bag_of_words,
        max_seq_len_in_batch=max_seq_len_in_batch
    )
    top_one_sequence = top_k_sequences[0]

    # # for debug only
    # bag_of_words =   [11, 12, 13, 25, 50, 198, 257, 262, 286, 287, 290, 318, 319, 338, 355, 357, 366, 373, 379, 383, 468,
    #                   471, 543, 550, 635, 810, 968, 1444, 1793, 1900, 2097, 2159, 2254, 2297, 2448, 3576, 3862, 3936, 4013,
    #                   4345, 4631, 5542, 7703, 7784, 8765, 9498, 11303, 11397, 11597, 11819, 12696, 12723, 13837, 14665, 15391,
    #                   25733, 29031, 45523]
    # top_one_sequence = [25733, 11, 262, 471, 13, 50, 13, 290, 262, 2159, 11, 290, 318, 635, 1900, 355]

    # Step 3: prior-based token reordering
    top_one_sequence = film_reordering(
        config=config,
        sequence=top_one_sequence,
        model=model,
        tokenizer=tokenizer,
        target_gradient=target_gradient,
        bag_of_words=bag_of_words
    )

    # Step 4: evaluate effectiveness
    recovered_data = (1, [top_one_sequence])  # need to "unsqueeze"
    post_processed_reconstructed_data = post_process(
        config=config,
        raw_data=recovered_data,
        model=model,
        tokenizer=tokenizer,
        is_embd=False
    )
    eval_attack(
        config=config,
        ground_truth_data=ground_truth_data,
        reconstructed_data=post_processed_reconstructed_data,
    )


'''
The code for gaining insight for determining the bag-of-word threshold:
    import matplotlib.pyplot as plt
    in_values = []
    for token in [383, 8765, 11819, 286, 262, 7703, 4631, 13837, 11, 635, 1900, 355, 471, 13, 50, 13]:
        s = torch.sum(torch.abs(text_embd_grad[token])).item()
        in_values.append(s)
    out_values = []
    for token in range(50257):
        if token in [383, 8765, 11819, 286, 262, 7703, 4631, 13837, 11, 635, 1900, 355, 471, 13, 50, 13]:
            continue
        s = torch.sum(torch.abs(text_embd_grad[token])).item()
        out_values.append(s)

    # plt.hist(in_values, bins=np.arange(min(in_values), max(in_values) + 0.2, 0.2), alpha=0.5, label='In')
    plt.hist(out_values, bins=np.arange(min(out_values), max(out_values) + 0.2, 0.2), alpha=0.5, label='Out')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title('Histogram of Sum Abs Embd')
    plt.legend()
    plt.show()
'''
