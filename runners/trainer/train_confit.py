# encoding: utf-8
from dataclasses import dataclass, field, asdict
from functools import partial
from transformers import AutoTokenizer, HfArgumentParser, set_seed
from torch.utils.data import DataLoader
from typing import Union
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from src.model.confit import (
    ConFitModel,
    ConFitModelArguments
)
from src.config.dataset import DATASET_CONFIG
from src.preprocess.dataset import RJPairNewContrastiveDataset, contrastive_rj_pair_w_indicies_collate_fn
from src.utils.test_embedding_networks import (
    EmbeddingNetworkTestArguments,
    load_test_data,
    get_metric_and_representations,
    evaluate,
)
from src.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint
import pytorch_lightning as pl
import pandas as pd
import torch
import wandb
import os
import sys
import shutil
import json


os.environ['TOKENIZERS_PARALLELISM'] = 'true'


@dataclass
class TrainingArguments:
    log_every_n_steps: int = field(
        default=10,
        metadata={"help": "Log every n steps."}
    )
    save_path: str = field(
        default="model_checkpoints/dual_encoder_contrastive/debug",
        metadata={"help": "The output directory."},
    )
    max_epochs: int = field(
        default=10,
        metadata={"help": "Max epochs."}
    )
    train_batch_size: int = field(
        default=16,
        metadata={"help": "Train batch size."}
    )
    val_batch_size: int = field(
        default=8,
        metadata={"help": "Validation batch size."}
    )
    strategy: str = field(
        default="ddp",
        metadata={"help": "Training strategy."},
    )
    precision: str = field(default='32', metadata={"help": "Precision."})
    log_group: str = field(
        default="aliyun",
        metadata={
            "help": "The name of the wandb run group to which the experiment belongs."
        },
    )
    seed: int = field(
        default=42,
        metadata={"help": "Random seed."}
    )
    no_save: bool = field(
        default=False,
        metadata={"help": "Whether to save the model."},
    )


@dataclass
class DataArguments:
    resume_data_path: str = field(
        default="dataset/AliTianChi/all_resume_w_updated_colnames.csv",
        metadata={"help": "Path to the resume data."},
    )
    job_data_path: str = field(
        default="dataset/AliTianChi/all_job_w_updated_colnames.csv",
        metadata={"help": "Path to the job data."},
    )
    query_prefix: str = field(
        default="",
        metadata={"help": "Prefix to add to query tokens."},
    )
    train_label_path: str = field(
        default="dataset/AliTianChi/train_labeled_data.jsonl",
        metadata={"help": "Path to the training labels."},
    )
    valid_label_path: str = field(
        default="dataset/AliTianChi/valid_classification_data.jsonl",
        metadata={"help": "Path to the validation labels."},
    )
    classification_data_path: str = field(
        default="dataset/AliTianChi/test_classification_data.jsonl",
        metadata={"help": "Path to the classification data for testing."},
    )
    rank_resume_data_path: str = field(
        default="dataset/AliTianChi/rank_resume.json",
        metadata={"help": "Path to the rank resume data for testing."},
    )
    rank_job_data_path: str = field(
        default="dataset/AliTianChi/rank_job.json",
        metadata={"help": "Path to the rank job data for testing."},
    )
    dataset_type: str = field(
        default="AliTianChi",
        metadata={"help": "The dataset type."},
    )
    num_hard_negatives: int = field(
        default=3,
        metadata={"help": "Maximum number of hard negatives per batch."},
    )
    # check if dataset type is valid
    def __post_init__(self):
        assert self.dataset_type in ["AliTianChi", "IntelliPro",], f"Invalid dataset type: {self.dataset_type}"
        assert self.dataset_type in self.resume_data_path, f"Datset type {self.dataset_type} does not match resume data path {self.resume_data_path}"
        assert self.dataset_type in self.job_data_path, f"Datset type {self.dataset_type} does not match job data path {self.job_data_path}"
        assert self.dataset_type in self.train_label_path, f"Datset type {self.dataset_type} does not match train label path {self.train_label_path}"
        assert self.dataset_type in self.valid_label_path, f"Datset type {self.dataset_type} does not match valid label path {self.valid_label_path}"
        assert self.dataset_type in self.classification_data_path, f"Datset type {self.dataset_type} does not match classification data path {self.classification_data_path}"
        assert self.dataset_type in self.rank_resume_data_path, f"Datset type {self.dataset_type} does not match rank resume data path {self.rank_resume_data_path}"
        assert self.dataset_type in self.rank_job_data_path, f"Datset type {self.dataset_type} does not match rank job data path {self.rank_job_data_path}"
        return


def get_dataloaders(
    model_args: ConFitModelArguments,
    trainer_args: TrainingArguments,
    data_args: DataArguments
):
    config = DATASET_CONFIG[data_args.dataset_type]
    max_seq_len_per_feature = config["max_seq_len_per_feature"]
    resume_taxon_token = config["resume_taxon_token"]
    job_taxon_token = config["job_taxon_token"]
    resume_key_names = config["resume_key_names"]
    job_key_names = config["job_key_names"]

    # sanity check
    num_resume_features = len(resume_key_names)
    num_job_features = len(job_key_names)
    assert model_args.num_resume_features == num_resume_features, f"{model_args.num_resume_features} != {num_resume_features}"
    assert model_args.num_job_features == num_job_features, f"{model_args.num_job_features} != {num_job_features}"

    # dataset
    all_resume_data = pd.read_csv(
        data_args.resume_data_path
    )
    all_job_data = pd.read_csv(
        data_args.job_data_path
    )
    training_labels: pd.DataFrame = pd.read_json(
        data_args.train_label_path, lines=True
    )
    validation_labels: pd.DataFrame = pd.read_json(
        data_args.valid_label_path, lines=True
    )

    all_resume_data_dict = all_resume_data.to_dict("records")
    all_job_data_dict = all_job_data.to_dict("records")
    training_labels_dict = training_labels.to_dict("records")
    validation_labels_dict = validation_labels.to_dict("records")

    ## dataset
    tokenizer = AutoTokenizer.from_pretrained(model_args.pretrained_encoder)

    train_dataset = RJPairNewContrastiveDataset(
        tokenizer,
        max_seq_length_per_key=max_seq_len_per_feature,
        resume_key_names=resume_key_names,
        job_key_names=job_key_names,
        tokenizer_args={
            "padding": "max_length",
            "return_tensors": "pt",
            "truncation": True,
        },
        all_resume_dict=all_resume_data_dict,
        all_job_dict=all_job_data_dict,
        label_pairs=training_labels_dict,
        resume_taxon_token=resume_taxon_token,
        job_taxon_token=job_taxon_token,
        query_prefix=data_args.query_prefix,
    )

    valid_dataset = RJPairNewContrastiveDataset(
        tokenizer,
        max_seq_length_per_key=max_seq_len_per_feature,
        resume_key_names=resume_key_names,
        job_key_names=job_key_names,
        tokenizer_args={
            "padding": "max_length",
            "return_tensors": "pt",
            "truncation": True,
        },
        all_resume_dict=all_resume_data_dict,
        all_job_dict=all_job_data_dict,
        label_pairs=validation_labels_dict,
        resume_taxon_token=resume_taxon_token,
        job_taxon_token=job_taxon_token,
        query_prefix=data_args.query_prefix,
    )

    train_dataloader = DataLoader(
        dataset=train_dataset,
        num_workers=4,
        batch_size=trainer_args.train_batch_size,
        shuffle=True,
        collate_fn=partial(
            contrastive_rj_pair_w_indicies_collate_fn,
            max_num_hard_negatives=data_args.num_hard_negatives,
        ),
        drop_last=True,  # makes things complicated IF len(batch) = 1 happens
    )
    valid_dataloader = DataLoader(
        dataset=valid_dataset,
        num_workers=4,
        batch_size=trainer_args.val_batch_size,
        shuffle=False,
        collate_fn=partial(
            contrastive_rj_pair_w_indicies_collate_fn,
            max_num_hard_negatives=data_args.num_hard_negatives,
        ),
    )
    return train_dataloader, valid_dataloader


def save_args(dataclass_args, save_path: str, save_name: str):
    os.makedirs(save_path, exist_ok=True)
    with open(os.path.join(save_path, save_name), "w", encoding='utf-8') as fwrite:
        json.dump(asdict(dataclass_args), fwrite, indent=4, sort_keys=True)
    return


def test_dual_encoder(model, model_args, trainer_args: TrainingArguments, data_args: DataArguments):
    print("Testing...")
    test_args = EmbeddingNetworkTestArguments(
        model_path=trainer_args.save_path,
        resume_data_path=data_args.resume_data_path,
        job_data_path=data_args.job_data_path,
        classification_validation_data_path=data_args.valid_label_path,
        classification_data_path=data_args.classification_data_path,
        rank_resume_data_path=data_args.rank_resume_data_path,
        rank_job_data_path=data_args.rank_job_data_path,
        dataset_type=data_args.dataset_type,
        query_prefix=data_args.query_prefix,
        batch_size=16,
        seed=trainer_args.seed,
    )
    all_test_data = load_test_data(test_args)
    (
        metric,
        test_rid_to_representation,
        test_jid_to_representation,
    ) = get_metric_and_representations(model, model_args, test_args, all_test_data)
    eval_results = evaluate(
        metric=metric,
        test_rid_to_representation=test_rid_to_representation,
        test_jid_to_representation=test_jid_to_representation,
        test_args=test_args,
    )

    eval_results_w_prefix = {}
    for k, v in eval_results.items():
        eval_results_w_prefix[f"test/{k}"] = v
    wandb.log(eval_results_w_prefix)
    return


def main(model_args: ConFitModelArguments, trainer_args: TrainingArguments, data_args: DataArguments):
    set_seed(trainer_args.seed)

    train_dataloader, valid_dataloader = get_dataloaders(model_args, trainer_args, data_args)

    ## logger
    run = wandb.init(
        project="resume",
        group=trainer_args.log_group,
        name=trainer_args.save_path.split("/")[-1],
        config={
            "model_args": asdict(model_args),
            "trainer_args": asdict(trainer_args),
            "data_args": asdict(data_args),
        },
    )
    wandb_logger = WandbLogger(
        experiment=run,  # the run you initialized via wandb.init
        resume=None,
        project="resume",
        log_model=False,
    )

    ## callbacks
    lr_monitor = LearningRateMonitor(logging_interval="step")
    checkpoint_callback = ModelCheckpoint(
        monitor="val/loss",
        mode="min",
        save_top_k=1,
        dirpath=trainer_args.save_path,
        every_n_epochs=1,
    )

    ## model and training
    # quick check
    if "offload" in trainer_args.strategy and model_args.cpu_adam == False:
        raise ValueError("cpu_adam must be True if offload is used.")
    model = ConFitModel(model_args)
    trainer = pl.Trainer(
        strategy=trainer_args.strategy,
        precision=trainer_args.precision,
        accelerator="gpu",
        devices=len(os.environ["CUDA_VISIBLE_DEVICES"].split(",")),
        logger=wandb_logger,
        log_every_n_steps=trainer_args.log_every_n_steps,
        default_root_dir=trainer_args.save_path,
        max_epochs=trainer_args.max_epochs,
        max_steps=trainer_args.max_epochs * len(train_dataloader),  # used by scheduler
        enable_checkpointing=True,
        callbacks=[lr_monitor, checkpoint_callback],
    )

    trainer.fit(model, train_dataloader, valid_dataloader)

    # load best model and test
    print('loading', checkpoint_callback.best_model_path)
    if "deepspeed" in trainer_args.strategy:
        wrapped_state_dict = get_fp32_state_dict_from_zero_checkpoint(checkpoint_callback.best_model_path)
        state_dict = {k.replace("_forward_module.", ""): v for k, v in wrapped_state_dict.items()}

        # clean up the zero folder
        output_file = checkpoint_callback.best_model_path + ".fp32"
        print(f"Saving fp32 state dict to {output_file}")
        torch.save({'state_dict': state_dict}, output_file)
        if os.path.exists(checkpoint_callback.best_model_path) and os.path.isdir(checkpoint_callback.best_model_path):
            shutil.rmtree(checkpoint_callback.best_model_path)
        checkpoint_callback.best_model_path = output_file
    else:
        state_dict = torch.load(checkpoint_callback.best_model_path)["state_dict"]
    model.load_state_dict(state_dict)
    model = model.float()  # ensures no bf16 in model
    model.eval()

    if trainer_args.no_save:
        # remove the checkpoint IFF it's a file
        if os.path.isfile(checkpoint_callback.best_model_path):
            os.remove(checkpoint_callback.best_model_path)
        else:
            print(f"[WARNING] Best model path {checkpoint_callback.best_model_path} is NOT a file, not removing it.")
    
    set_seed(trainer_args.seed)
    test_dual_encoder(model, model_args, trainer_args, data_args)
    return


if __name__ == "__main__":
    parser = HfArgumentParser(
		dataclass_types=(ConFitModelArguments, TrainingArguments, DataArguments),
		description="resume matching"
	)
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, trainer_args, data_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    else:
        model_args, trainer_args, data_args = parser.parse_args_into_dataclasses()
    print('received model_args:')
    print(json.dumps(asdict(model_args), indent=2, sort_keys=True))
    print('received trainer_args:')
    print(json.dumps(asdict(trainer_args), indent=2, sort_keys=True))
    print('received data_args:')
    print(json.dumps(asdict(data_args), indent=2, sort_keys=True))

    # save configs
    save_args(model_args, trainer_args.save_path, 'model_args.json')
    save_args(trainer_args, trainer_args.save_path, 'trainer_args.json')
    save_args(data_args, trainer_args.save_path, 'data_args.json')
    
    # main training code
    main(model_args, trainer_args, data_args)