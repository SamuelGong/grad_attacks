import os
import yaml
from datetime import datetime
from munch import DefaultMunch  # nested dict to object


def load_configurations(config_file):
    assert os.path.isfile(config_file)
    with open(config_file, 'rb') as fin:
        config_dict = yaml.load(fin, Loader=yaml.FullLoader)

    if config_dict["datasource"]["dataset"] == "wikitext-2":
        dataset_full_name = ("wikitext", "wikitext-2-raw-v1")
        main_column = "text"  # for removing input of zero length
        unused_columns_to_remove = [main_column]
    elif config_dict["datasource"]["dataset"] == "cola":
        dataset_full_name = ("glue", "cola")
        main_column = "sentence"
        label_column = "label"
        other_columns = ["idx"]
        unused_columns_to_remove = [main_column] + other_columns
        # label is not to remove unless it is for text generation
    else:
        raise NotImplementedError

    datasource_update_dict = {
        "full_name": dataset_full_name,
        "main_column": main_column,
        "unused_columns_to_remove": unused_columns_to_remove
    }
    if config_dict["task"] == "text-classification":
        datasource_update_dict["label_column"] = label_column
    config_dict["datasource"].update(datasource_update_dict)

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    config_dict.update({
        # "gradient_file": config_file.replace(".yml", "_grad.bin"),
        "log_file": config_file.replace(".yml", f"_{timestamp}.log")
    })

    # print(config_dict)
    result = DefaultMunch.fromDict(config_dict)
    return result
