import json
import logging
from evaluate import load


def eval_attack(config, ground_truth_data, reconstructed_data):
    report_dict = {}
    if config.task in ["text-generation", "text-classification"]:
        logging.info(f'Reconstructed text: {reconstructed_data["text"]}')
        if config.task == "text-classification":
            logging.info(f'Reconstructed label: {reconstructed_data["label"]}')

        for metric in config.eval.metrics:
            scorer = load(metric)
            if metric in ["bleu", "rouge"]:
                results = scorer.compute(
                    predictions=reconstructed_data["text"],
                    references=ground_truth_data["text"]
                )
                report_dict[metric] = results
            elif metric in ["accuracy"]:
                for rec_token, true_token in zip(reconstructed_data["token"], ground_truth_data["token"]):
                    if config.attack.name == "film":
                        # the token list length may not match
                        min_len = min(len(true_token), len(rec_token))
                        rec_token = rec_token[:min_len]
                        true_token = true_token[:min_len]
                    scorer.add_batch(
                        predictions=rec_token,
                        references=true_token
                    )
                report_dict[metric] = scorer.compute()[metric]
    else:
        raise NotImplementedError

    logging.info(f'Metric report: {json.dumps(report_dict)}')
    # logging.info(f'Metric report: {json.dumps(report_dict, indent=4)}')
