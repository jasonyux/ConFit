from transformers import HfArgumentParser, set_seed
from dataclasses import dataclass, field, asdict
from src.model.inexit import InEXITModelArguments, InEXITModel
from src.utils.test_score_networks import (
    ScoreNetworkTestArguments,
    load_test_data,
    get_metric_and_representations_for_inexit,
    evaluate,
    evaluate_speed_for_inexit
)
import torch
import os
import json
import sys
import wandb


os.environ["TOKENIZERS_PARALLELISM"] = "true"


@dataclass
class ModelArguments:
    model_checkpoint_name: str = field(
        default="epoch=3-step=5532.ckpt",
        metadata={"help": "Name of the model checkpoint file"},
    )


@dataclass
class LoggingArguments:
    wandb_run_id: str = field(
        default="",
        metadata={"help": "wandb run id to log to. If empty, will not log to wandb"},
    )


def load_model(test_args: ScoreNetworkTestArguments, model_args: ModelArguments):
    model_args_file = os.path.join(test_args.model_path, "model_args.json")
    with open(model_args_file, "r", encoding="utf-8") as fread:
        model_args_dict = json.load(fread)

    loaded_model_args = InEXITModelArguments(**model_args_dict)

    model = InEXITModel(loaded_model_args)

    checkpoint_path = os.path.join(
        test_args.model_path, model_args.model_checkpoint_name
    )
    model.load_state_dict(torch.load(checkpoint_path)["state_dict"])
    return model, loaded_model_args


def main(
    test_args: ScoreNetworkTestArguments,
    model_args: ModelArguments,
    logging_args: LoggingArguments,
):
    set_seed(test_args.seed)
    all_test_data = load_test_data(test_args)
    model, loaded_model_args = load_model(test_args, model_args)
    
    (
        metric,
        test_rid_to_representation,
        test_jid_to_representation,
    ) = get_metric_and_representations_for_inexit(
        model, loaded_model_args, test_args, all_test_data
    )

    eval_results = evaluate(
        metric=metric,
        test_rid_to_representation=test_rid_to_representation,
        test_jid_to_representation=test_jid_to_representation,
        test_args=test_args,
    )

    eval_speed = evaluate_speed_for_inexit(
        model, loaded_model_args, test_args, all_test_data,
    )

    if logging_args.wandb_run_id != "":
        wandb.init(
            project="resume",
            id=logging_args.wandb_run_id,
            resume="must",
        )
        eval_results_w_prefix = {}
        for k, v in eval_results.items():
            eval_results_w_prefix[f"test/{k}"] = v
        for k, v in eval_speed.items():
            eval_results_w_prefix[f"test/{k}"] = v
        wandb.log(eval_results_w_prefix)
        wandb.finish()
    return


if __name__ == "__main__":
    parser = HfArgumentParser(
        dataclass_types=(ScoreNetworkTestArguments, ModelArguments, LoggingArguments),
        description="resume matching testing",
    )
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        test_args, model_args, logging_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    else:
        test_args, model_args, logging_args = parser.parse_args_into_dataclasses()
    print("received test_args:")
    print(json.dumps(asdict(test_args), indent=2, sort_keys=True))
    print("received model_args:")
    print(json.dumps(asdict(model_args), indent=2, sort_keys=True))
    print("received logging_args:")
    print(json.dumps(asdict(logging_args), indent=2, sort_keys=True))

    main(test_args, model_args, logging_args)
