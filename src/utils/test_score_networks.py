from transformers import AutoTokenizer
from tqdm.auto import tqdm
from torch.utils.data import DataLoader, default_collate
from dataclasses import dataclass, field
from src.evaluation.metrics import PrecomputedMetric, Metric
from src.evaluation.eval import EvalFindClassificationThreshold, EvalClassification, EvalRanking
from src.config.dataset import DATASET_CONFIG
from src.preprocess.dataset import (
    RJPairDataset, RJPairSimplifiedDataset, RJPair2DSimplifiedDataset,
    rj_pair_collate_fn
)
import pandas as pd
import pickle
import torch
import os
import json
import time


@dataclass
class ScoreNetworkTestArguments:
    model_path: str = field(
        default="model_checkpoints/InEXIT/aliyun_1e-9_decay",
        metadata={
            "help": "Path to the model we are evaluating and will dump results there."
        },
    )
    resume_data_path: str = field(
        default="dataset/AliTianChi/all_resume_w_updated_colnames.csv",
        metadata={"help": "Path to the resume data."},
    )
    job_data_path: str = field(
        default="dataset/AliTianChi/all_job_w_updated_colnames.csv",
        metadata={"help": "Path to the job data."},
    )
    classification_validation_data_path: str = field(
        default="dataset/AliTianChi/valid_classification_data.jsonl",
        metadata={"help": "Path to the classification data."},
    )
    classification_data_path: str = field(
        default="dataset/AliTianChi/test_classification_data.jsonl",
        metadata={"help": "Path to the classification data."},
    )
    rank_resume_data_path: str = field(
        default="dataset/AliTianChi/rank_resume.json",
        metadata={"help": "Path to the rank resume data."},
    )
    rank_job_data_path: str = field(
        default="dataset/AliTianChi/rank_job.json",
        metadata={"help": "Path to the rank job data."},
    )
    dataset_type: str = field(
        default="AliTianChi",
        metadata={"help": "Type of the dataset."},
    )
    query_prefix: str = field(
        default="",
        metadata={"help": "Query prefix to use."},
    )
    batch_size: int = field(
        default=32,
        metadata={"help": "Batch size for testing"},
    )
    seed: int = field(
        default=42,
        metadata={"help": "Seed for testing"},
    )

    def __post_init__(self):
        assert self.dataset_type in [
            "AliTianChi",
            "IntelliPro",
        ], f"Invalid dataset type: {self.dataset_type}"
        assert (
            self.dataset_type in self.resume_data_path
        ), f"Datset type {self.dataset_type} does not match resume data path {self.resume_data_path}"
        assert (
            self.dataset_type in self.job_data_path
        ), f"Datset type {self.dataset_type} does not match job data path {self.job_data_path}"
        assert (
            self.dataset_type in self.classification_data_path
        ), f"Datset type {self.dataset_type} does not match classification_data_path {self.classification_data_path}"
        assert (
            self.dataset_type in self.rank_resume_data_path
        ), f"Datset type {self.dataset_type} does not match rank_resume_data_path {self.rank_resume_data_path}"
        assert (
            self.dataset_type in self.rank_job_data_path
        ), f"Datset type {self.dataset_type} does not match rank_job_data_path {self.rank_job_data_path}"
        return


def load_test_data(test_args: ScoreNetworkTestArguments):
    all_resume_data = pd.read_csv(test_args.resume_data_path)
    all_job_data = pd.read_csv(test_args.job_data_path)
    all_resume_data_dict = all_resume_data.to_dict("records")
    all_job_data_dict = all_job_data.to_dict("records")

    all_test_pairs_to_predict = []
    seen_pairs = set()

    test_classification_labels = pd.read_json(
        test_args.classification_data_path, lines=True
    )
    valid_classification_labels = pd.read_json(
        test_args.classification_validation_data_path, lines=True
    )
    test_ranking_resume_labels = json.load(open(test_args.rank_resume_data_path))
    test_ranking_job_labels = json.load(open(test_args.rank_job_data_path))

    for i, row in valid_classification_labels.iterrows():
        user_id = row["user_id"]
        jd_no = row["jd_no"]
        satisfied = row["satisfied"]
        identifier = f"{user_id}-{jd_no}"
        if identifier in seen_pairs:
            continue
        all_test_pairs_to_predict.append(
            {"user_id": str(user_id), "jd_no": str(jd_no), "satisfied": int(satisfied)}
        )
        seen_pairs.add(identifier)
    
    for i, row in test_classification_labels.iterrows():
        user_id = row["user_id"]
        jd_no = row["jd_no"]
        satisfied = row["satisfied"]
        identifier = f"{user_id}-{jd_no}"
        if identifier in seen_pairs:
            continue
        all_test_pairs_to_predict.append(
            {"user_id": str(user_id), "jd_no": str(jd_no), "satisfied": int(satisfied)}
        )
        seen_pairs.add(identifier)

    for jd_no, data in test_ranking_resume_labels.items():
        user_ids = data["user_ids"]
        satisfieds = data["satisfied"]
        for user_id, satisfied in zip(user_ids, satisfieds):
            identifier = f"{user_id}-{jd_no}"
            if identifier in seen_pairs:
                continue
            all_test_pairs_to_predict.append(
                {
                    "user_id": str(user_id),
                    "jd_no": str(jd_no),
                    "satisfied": int(satisfied),
                }
            )
            seen_pairs.add(identifier)

    for user_id, data in test_ranking_job_labels.items():
        jd_nos = data["jd_nos"]
        satisfieds = data["satisfied"]
        for jd_no, satisfied in zip(jd_nos, satisfieds):
            identifier = f"{user_id}-{jd_no}"
            if identifier in seen_pairs:
                continue
            all_test_pairs_to_predict.append(
                {
                    "user_id": str(user_id),
                    "jd_no": str(jd_no),
                    "satisfied": int(satisfied),
                }
            )
            seen_pairs.add(identifier)

    print(f"Total number of test pairs to compute: {len(all_test_pairs_to_predict)}")
    return {
        "all_resume_data_dict": all_resume_data_dict,
        "all_job_data_dict": all_job_data_dict,
        "all_test_pairs_to_predict": all_test_pairs_to_predict,
    }


def get_metric_and_representations_for_inexit(
    model, model_args, test_args: ScoreNetworkTestArguments, all_test_data: dict
) -> Metric:
    """assumes that the following would work:
    ```python
    output = model.predict_step(batch, batch_idx)
    class_1_prob = torch.sigmoid(output.logits)
    ```
    and assumes your test dataset loads by:
    ```python
    RJPairDataset
    ```
    """
    config = DATASET_CONFIG[test_args.dataset_type]
    max_seq_len_per_feature = config["max_seq_len_per_feature"]
    max_key_seq_length = config["max_key_seq_length"]
    resume_taxon_token = config["resume_taxon_token"]
    job_taxon_token = config["job_taxon_token"]
    resume_key_names = config["resume_key_names"]
    job_key_names = config["job_key_names"]
    query_prefix = test_args.query_prefix

    tokenizer = AutoTokenizer.from_pretrained(model_args.pretrained_encoder)

    test_dataset = RJPairDataset(
        tokenizer,
        max_key_seq_length=max_key_seq_length,
        max_seq_length_per_key=max_seq_len_per_feature,
        resume_key_names=resume_key_names,
        job_key_names=job_key_names,
        tokenizer_args={
            "padding": "max_length",
            "return_tensors": "pt",
            "truncation": True,
        },
        all_resume_dict=all_test_data["all_resume_data_dict"],
        all_job_dict=all_test_data["all_job_data_dict"],
        label_pairs=all_test_data["all_test_pairs_to_predict"],
        resume_taxon_token=resume_taxon_token,
        job_taxon_token=job_taxon_token,
        query_prefix=query_prefix,
    )

    test_dataloader = DataLoader(
        dataset=test_dataset,
        num_workers=4,
        batch_size=test_args.batch_size,
        shuffle=False,
        collate_fn=rj_pair_collate_fn,
    )

    # precomputing the scores
    all_test_pairs_to_predict_scored = {}

    model = model.to("cuda")
    model.eval()

    for batch_idx, batch in tqdm(
        enumerate(test_dataloader), total=len(test_dataloader)
    ):
        base_idx = batch_idx * test_args.batch_size
        with torch.no_grad():
            output = model.predict_step(batch, batch_idx)
            # sigmoid output is used more often in baselines
            # class_1_prob = torch.nn.functional.softmax(output.logits, dim=1)[:, 1]
            class_1_prob = torch.sigmoid(output.logits)

        for offset, prob in enumerate(class_1_prob):
            pair = all_test_data["all_test_pairs_to_predict"][base_idx + offset]
            user_id = pair["user_id"]
            jd_no = pair["jd_no"]

            all_test_pairs_to_predict_scored[(user_id, jd_no)] = float(
                prob.cpu().numpy()
            )

    score_file_save_path = os.path.join(test_args.model_path, "test_score.pkl")
    with open(score_file_save_path, "wb") as fwrite:
        pickle.dump(all_test_pairs_to_predict_scored, fwrite)

    # scoring
    metric = PrecomputedMetric(
        precomputed_scores=all_test_pairs_to_predict_scored,
    )
    return metric, {}, {}  # empty because we are doing a precomputed metric


def evaluate_speed_for_inexit(model, model_args, test_args: ScoreNetworkTestArguments, all_test_data: dict):
    # initialization: compute embeddings
    config = DATASET_CONFIG[test_args.dataset_type]
    max_seq_len_per_feature = config["max_seq_len_per_feature"]
    max_key_seq_length = config["max_key_seq_length"]
    resume_taxon_token = config["resume_taxon_token"]
    job_taxon_token = config["job_taxon_token"]
    resume_key_names = config["resume_key_names"]
    job_key_names = config["job_key_names"]
    query_prefix = test_args.query_prefix

    tokenizer = AutoTokenizer.from_pretrained(model_args.pretrained_encoder)

    if not next(model.parameters()).is_cuda:
        model = model.to("cuda")
    model.eval()
    
    # rank 100, 1000, and 10000 (or all) jobs
    stats_dict = {}
    num_jobs_to_rank = [100, 1000, 10000]
    for n_j_to_rank in num_jobs_to_rank:
        ### initialization: get the embeddings
        # we don't need to actually get the result, just measure the speed
        print(f"Ranking {n_j_to_rank} (fake) jobs")

        num_to_actually_embed = 100
        assert n_j_to_rank % num_to_actually_embed == 0

        test_dataset = RJPairDataset(
            tokenizer,
            max_key_seq_length=max_key_seq_length,
            max_seq_length_per_key=max_seq_len_per_feature,
            resume_key_names=resume_key_names,
            job_key_names=job_key_names,
            tokenizer_args={
                "padding": "max_length",
                "return_tensors": "pt",
                "truncation": True,
            },
            all_resume_dict=all_test_data["all_resume_data_dict"],
            all_job_dict=all_test_data["all_job_data_dict"],
            label_pairs=all_test_data["all_test_pairs_to_predict"][:num_to_actually_embed],
            resume_taxon_token=resume_taxon_token,
            job_taxon_token=job_taxon_token,
            query_prefix=query_prefix,
        )

        test_dataloader = DataLoader(
            dataset=test_dataset,
            num_workers=4,
            batch_size=test_args.batch_size,
            shuffle=False,
            collate_fn=rj_pair_collate_fn,
        )

        # here to score, we ACTUALLY NEED TO RUN THE ENTIRE NETWORK
        # therefore, runtime is scoring EVERY PAIR
        _search_start = time.time()
        for batch_idx, batch in tqdm(
            enumerate(test_dataloader), total=len(test_dataloader)
        ):
            with torch.no_grad():
                output = model.predict_step(batch, batch_idx)

        _search_spent = time.time() - _search_start
        _search_spent *= (n_j_to_rank / num_to_actually_embed)
        _search_spent /= 2  # assuming we are ranking the SAME resume against all jobs, so resume can be reused
        print(f"Search time spent: {_search_spent:.2f} seconds for {n_j_to_rank=}, batch_size={test_args.batch_size}")
        ## need to manually minus the embedding time (use the initialization time for ConFit)

        ## store stats
        stats_dict[f'rank{n_j_to_rank}_init'] = -1.0  # none
        stats_dict[f'rank{n_j_to_rank}_search'] = _search_spent
        stats_dict[f'rank{n_j_to_rank}_bsz'] = test_args.batch_size
    return stats_dict


def get_metric_and_representations_for_mvcon(
    model, model_args, test_args: ScoreNetworkTestArguments, all_test_data: dict
) -> Metric:
    """assumes that the following would work:
    ```python
    output = model.predict_step(batch, batch_idx)
    class_1_prob = torch.sigmoid(output.logits)
    ```
    and dataset is
    ```
    RJPair2DSimplifiedDataset
    ```
    """
    config = DATASET_CONFIG[test_args.dataset_type]
    resume_taxon_token = config["resume_taxon_token"]
    job_taxon_token = config["job_taxon_token"]
    resume_key_names = config["resume_key_names"]
    job_key_names = config["job_key_names"]
    query_prefix = test_args.query_prefix

    tokenizer = AutoTokenizer.from_pretrained(model_args.pretrained_encoder)

    test_dataset = RJPair2DSimplifiedDataset(
        tokenizer,
        max_seq_len=256,
        resume_key_names=resume_key_names,
        job_key_names=job_key_names,
        tokenizer_args={
            "padding": "max_length",
            "return_tensors": "pt",
            "truncation": True,
        },
        all_resume_dict=all_test_data["all_resume_data_dict"],
        all_job_dict=all_test_data["all_job_data_dict"],
        label_pairs=all_test_data["all_test_pairs_to_predict"],
        resume_taxon_token=resume_taxon_token,
        job_taxon_token=job_taxon_token,
        query_prefix=query_prefix,
    )

    test_dataloader = DataLoader(
        dataset=test_dataset,
        num_workers=4,
        batch_size=test_args.batch_size,
        shuffle=False,
        collate_fn=default_collate,
    )

    # precomputing the scores
    all_test_pairs_to_predict_scored = {}

    model = model.to("cuda")
    model.eval()

    for batch_idx, batch in tqdm(
        enumerate(test_dataloader), total=len(test_dataloader)
    ):
        base_idx = batch_idx * test_args.batch_size
        with torch.no_grad():
            output = model.predict_step(batch, batch_idx)
            # sigmoid output is used more often in baselines
            # class_1_prob = torch.nn.functional.softmax(output.logits, dim=1)[:, 1]
            class_1_prob = torch.sigmoid(output.logits)

        for offset, prob in enumerate(class_1_prob):
            pair = all_test_data["all_test_pairs_to_predict"][base_idx + offset]
            user_id = pair["user_id"]
            jd_no = pair["jd_no"]

            all_test_pairs_to_predict_scored[(user_id, jd_no)] = float(
                prob.cpu().numpy()
            )

    score_file_save_path = os.path.join(test_args.model_path, "test_score.pkl")
    with open(score_file_save_path, "wb") as fwrite:
        pickle.dump(all_test_pairs_to_predict_scored, fwrite)

    # scoring
    metric = PrecomputedMetric(
        precomputed_scores=all_test_pairs_to_predict_scored,
    )
    return metric, {}, {}  # empty because we are doing a precomputed metric


def evaluate_speed_for_mvcon(model, model_args, test_args: ScoreNetworkTestArguments, all_test_data: dict):
    # initialization: compute embeddings
    config = DATASET_CONFIG[test_args.dataset_type]
    resume_taxon_token = config["resume_taxon_token"]
    job_taxon_token = config["job_taxon_token"]
    resume_key_names = config["resume_key_names"]
    job_key_names = config["job_key_names"]
    query_prefix = test_args.query_prefix

    tokenizer = AutoTokenizer.from_pretrained(model_args.pretrained_encoder)

    if not next(model.parameters()).is_cuda:
        model = model.to("cuda")
    model.eval()
    
    # rank 100, 1000, and 10000 (or all) jobs
    stats_dict = {}
    num_jobs_to_rank = [100, 1000, 10000]
    for n_j_to_rank in num_jobs_to_rank:
        ### initialization: get the embeddings
        # we don't need to actually get the result, just measure the speed
        print(f"Ranking {n_j_to_rank} (fake) jobs")

        num_to_actually_embed = 100
        assert n_j_to_rank % num_to_actually_embed == 0

        test_dataset = RJPair2DSimplifiedDataset(
            tokenizer,
            max_seq_len=256,
            resume_key_names=resume_key_names,
            job_key_names=job_key_names,
            tokenizer_args={
                "padding": "max_length",
                "return_tensors": "pt",
                "truncation": True,
            },
            all_resume_dict=all_test_data["all_resume_data_dict"],
            all_job_dict=all_test_data["all_job_data_dict"],
            label_pairs=all_test_data["all_test_pairs_to_predict"][:num_to_actually_embed],
            resume_taxon_token=resume_taxon_token,
            job_taxon_token=job_taxon_token,
            query_prefix=query_prefix,
        )

        test_dataloader = DataLoader(
            dataset=test_dataset,
            num_workers=4,
            batch_size=test_args.batch_size,
            shuffle=False,
            collate_fn=default_collate,
        )

        # here to score, we ACTUALLY NEED TO RUN THE LAST PART OF THE NETWORK
        # therefore, runtime is scoring EVERY PAIR
        _search_start = time.time()
        for batch_idx, batch in tqdm(
            enumerate(test_dataloader), total=len(test_dataloader)
        ):
            with torch.no_grad():
                output = model.predict_step(batch, batch_idx)
                # sigmoid output is used more often in baselines
                # class_1_prob = torch.nn.functional.softmax(output.logits, dim=1)[:, 1]
                class_1_prob = torch.sigmoid(output.logits)

        _search_spent = time.time() - _search_start
        _search_spent *= (n_j_to_rank / num_to_actually_embed)
        _search_spent /= 4  # assuming we are ranking the SAME resume against all jobs, so resume can be reused (and MVCON computes both views)
        print(f"Search time spent: {_search_spent:.2f} seconds for {n_j_to_rank=}, batch_size={test_args.batch_size}")
        ## need to manually minus the embedding time (use the initialization time for ConFit)

        ## store stats
        stats_dict[f'rank{n_j_to_rank}_init'] = -1.0  # none
        stats_dict[f'rank{n_j_to_rank}_search'] = _search_spent
        stats_dict[f'rank{n_j_to_rank}_bsz'] = test_args.batch_size
    return stats_dict


def get_metric_and_representations_for_dpgnn(
    model, model_args, test_args: ScoreNetworkTestArguments, all_test_data: dict
) -> Metric:
    """assumes that the following would work:
    ```python
    output = model.predict_step(batch, batch_idx)
    class_1_prob = torch.sigmoid(output.logits)
    ```
    and dataset is
    ```
    RJPair2DSimplifiedDataset
    ```
    """
    config = DATASET_CONFIG[test_args.dataset_type]
    max_seq_len_per_feature = config["max_seq_len_per_feature"]
    resume_taxon_token = config["resume_taxon_token"]
    job_taxon_token = config["job_taxon_token"]
    resume_key_names = config["resume_key_names"]
    job_key_names = config["job_key_names"]
    query_prefix = test_args.query_prefix

    tokenizer = AutoTokenizer.from_pretrained(model_args.pretrained_encoder)

    test_dataset = RJPairSimplifiedDataset(
        tokenizer,
        max_seq_length_per_key=max_seq_len_per_feature,
        resume_key_names=resume_key_names,
        job_key_names=job_key_names,
        tokenizer_args={
            "padding": "max_length",
            "return_tensors": "pt",
            "truncation": True,
        },
        all_resume_dict=all_test_data["all_resume_data_dict"],
        all_job_dict=all_test_data["all_job_data_dict"],
        label_pairs=all_test_data["all_test_pairs_to_predict"],
        resume_taxon_token=resume_taxon_token,
        job_taxon_token=job_taxon_token,
        query_prefix=query_prefix,
    )

    test_dataloader = DataLoader(
        dataset=test_dataset,
        num_workers=4,
        batch_size=test_args.batch_size,
        shuffle=False,
        collate_fn=rj_pair_collate_fn,
    )

    # precomputing the scores
    all_test_pairs_to_predict_scored = {}
    rid_to_representation = {}
    jid_to_representation = {}

    model = model.to("cuda")
    model.eval()

    for batch_idx, batch in tqdm(
        enumerate(test_dataloader), total=len(test_dataloader)
    ):
        base_idx = batch_idx * test_args.batch_size
        with torch.no_grad():
            output = model.predict_step(batch, batch_idx)
            # sigmoid output is used more often in baselines
            # class_1_prob = torch.nn.functional.softmax(output.logits, dim=1)[:, 1]
            class_1_prob = torch.sigmoid(output.logits)
        # save the penultimate embeddings as well
        batched_resume_representation = output.batched_resume_representation
        batched_job_representation = output.batched_job_representation

        for offset, prob in enumerate(class_1_prob):
            pair = all_test_data["all_test_pairs_to_predict"][base_idx + offset]
            user_id = pair["user_id"]
            jd_no = pair["jd_no"]

            all_test_pairs_to_predict_scored[(user_id, jd_no)] = float(
                prob.cpu().numpy()
            )

            rid_to_representation[user_id] = batched_resume_representation[offset].cpu().numpy()
            jid_to_representation[jd_no] = batched_job_representation[offset].cpu().numpy()

    embedding_file_save_path = os.path.join(test_args.model_path, "test_embedding.pkl")
    with open(embedding_file_save_path, "wb") as fwrite:
        pickle.dump(
            {
                "rid_to_representation": rid_to_representation,
                "jid_to_representation": jid_to_representation,
            },
            fwrite,
        )

    score_file_save_path = os.path.join(test_args.model_path, "test_score.pkl")
    with open(score_file_save_path, "wb") as fwrite:
        pickle.dump(all_test_pairs_to_predict_scored, fwrite)

    # scoring
    metric = PrecomputedMetric(
        precomputed_scores=all_test_pairs_to_predict_scored,
    )
    return metric, {}, {}  # empty because we are doing a precomputed metric


def evaluate_speed_for_dpgnn(model, model_args, test_args: ScoreNetworkTestArguments, all_test_data: dict):
    # initialization: compute embeddings
    config = DATASET_CONFIG[test_args.dataset_type]
    resume_taxon_token = config["resume_taxon_token"]
    job_taxon_token = config["job_taxon_token"]
    resume_key_names = config["resume_key_names"]
    job_key_names = config["job_key_names"]
    query_prefix = test_args.query_prefix

    tokenizer = AutoTokenizer.from_pretrained(model_args.pretrained_encoder)

    if not next(model.parameters()).is_cuda:
        model = model.to("cuda")
    model.eval()
    
    # rank 100, 1000, and 10000 (or all) jobs
    stats_dict = {}
    num_jobs_to_rank = [100, 1000, 10000]
    for n_j_to_rank in num_jobs_to_rank:
        ### initialization: get the embeddings
        # we don't need to actually get the result, just measure the speed
        print(f"Ranking {n_j_to_rank} (fake) jobs")

        num_to_actually_embed = 100
        assert n_j_to_rank % num_to_actually_embed == 0

        test_dataset = RJPair2DSimplifiedDataset(
            tokenizer,
            max_seq_len=256,
            resume_key_names=resume_key_names,
            job_key_names=job_key_names,
            tokenizer_args={
                "padding": "max_length",
                "return_tensors": "pt",
                "truncation": True,
            },
            all_resume_dict=all_test_data["all_resume_data_dict"],
            all_job_dict=all_test_data["all_job_data_dict"],
            label_pairs=all_test_data["all_test_pairs_to_predict"][:num_to_actually_embed],
            resume_taxon_token=resume_taxon_token,
            job_taxon_token=job_taxon_token,
            query_prefix=query_prefix,
        )

        test_dataloader = DataLoader(
            dataset=test_dataset,
            num_workers=4,
            batch_size=test_args.batch_size,
            shuffle=False,
            collate_fn=default_collate,
        )

        # here to score, we ACTUALLY NEED TO RUN THE LAST PART OF THE NETWORK
        # therefore, runtime is scoring EVERY PAIR
        _search_start = time.time()
        for batch_idx, batch in tqdm(
            enumerate(test_dataloader), total=len(test_dataloader)
        ):
            with torch.no_grad():
                output = model.predict_step(batch, batch_idx)
                # sigmoid output is used more often in baselines
                # class_1_prob = torch.nn.functional.softmax(output.logits, dim=1)[:, 1]
                class_1_prob = torch.sigmoid(output.logits)

        _search_spent = time.time() - _search_start
        _search_spent *= (n_j_to_rank / num_to_actually_embed)
        _search_spent /= 2  # assuming we are ranking the SAME resume against all jobs, and we only need the passive embeddings
        print(f"Search time spent: {_search_spent:.2f} seconds for {n_j_to_rank=}, batch_size={test_args.batch_size}")
        ## need to manually minus the embedding time (use the initialization time for ConFit)

        ## store stats
        stats_dict[f'rank{n_j_to_rank}_init'] = -1.0  # none
        stats_dict[f'rank{n_j_to_rank}_search'] = _search_spent
        stats_dict[f'rank{n_j_to_rank}_bsz'] = test_args.batch_size
    return stats_dict


def evaluate(
    metric: Metric,
    test_rid_to_representation: dict,
    test_jid_to_representation: dict,
    test_args: ScoreNetworkTestArguments,
):
    all_results = {}

    ## classification
    # first find threshold
    valid_labels = pd.read_json(test_args.classification_validation_data_path, lines=True)
    valid_labels_dict = valid_labels.to_dict("records")
    valid_labels_tuple = []
    for pair in valid_labels_dict:
        user_id = str(pair["user_id"])
        jd_no = str(pair["jd_no"])
        label = int(pair["satisfied"])
        valid_labels_tuple.append((user_id, jd_no, label))
    
    evaluator = EvalFindClassificationThreshold(
        metric,
        test_rid_to_representation=test_rid_to_representation,
        test_jid_to_representation=test_jid_to_representation,
        test_pairs=valid_labels_tuple,
        offline_mode=True,
    )
    score_report, _ = evaluator.evaluate()
    best_threshold = score_report["best_threshold"]

    ### Test: classification
    test_labels = pd.read_json(test_args.classification_data_path, lines=True)
    test_labels_dict = test_labels.to_dict("records")
    test_labels_tuple = []
    for pair in test_labels_dict:
        user_id = str(pair["user_id"])
        jd_no = str(pair["jd_no"])
        label = int(pair["satisfied"])
        test_labels_tuple.append((user_id, jd_no, label))

    evaluator = EvalClassification(
        metric,
        test_rid_to_representation=test_rid_to_representation,
        test_jid_to_representation=test_jid_to_representation,
        test_pairs=test_labels_tuple,
        threshold=best_threshold,
        offline_mode=True,  # ScoreNetworks produces scores e2e, so we need to use precomputed scores
    )

    score_report, eval_history = evaluator.evaluate()
    all_results = {**all_results, **score_report}

    print(json.dumps(score_report, indent=4))

    with open(
        os.path.join(test_args.model_path, "classification_score_report.json"),
        "w",
        encoding="utf-8",
    ) as fwrite:
        json.dump(score_report, fwrite, indent=4)
    with open(
        os.path.join(test_args.model_path, "classification_eval_history.pkl"), "wb"
    ) as fwrite:
        pickle.dump(eval_history, fwrite)

    ## rank job
    test_ranking_labels = json.load(
        open(test_args.rank_job_data_path, encoding="utf-8")
    )
    evaluator = EvalRanking(
        metric,
        test_rid_to_representation=test_rid_to_representation,
        test_jid_to_representation=test_jid_to_representation,
        test_ranking_data=test_ranking_labels,
        offline_mode=True,
    )

    score_report, eval_history = evaluator.evaluate()
    for k, v in score_report.items():
        all_results[f"rank_job_{k}"] = v

    print(json.dumps(score_report, indent=4))

    with open(
        os.path.join(test_args.model_path, "rank_job_score_report.json"),
        "w",
        encoding="utf-8",
    ) as fwrite:
        json.dump(score_report, fwrite, indent=4)
    with open(
        os.path.join(test_args.model_path, "rank_job_eval_history.pkl"), "wb"
    ) as fwrite:
        pickle.dump(eval_history, fwrite)

    ## rank resume
    test_ranking_labels = json.load(
        open(test_args.rank_resume_data_path, encoding="utf-8")
    )
    evaluator = EvalRanking(
        metric,
        test_rid_to_representation=test_rid_to_representation,
        test_jid_to_representation=test_jid_to_representation,
        test_ranking_data=test_ranking_labels,
        offline_mode=True,
    )

    score_report, eval_history = evaluator.evaluate()
    for k, v in score_report.items():
        all_results[f"rank_resume_{k}"] = v

    print(json.dumps(score_report, indent=4))

    with open(
        os.path.join(test_args.model_path, "rank_resume_score_report.json"),
        "w",
        encoding="utf-8",
    ) as fwrite:
        json.dump(score_report, fwrite, indent=4)
    with open(
        os.path.join(test_args.model_path, "rank_resume_eval_history.pkl"), "wb"
    ) as fwrite:
        pickle.dump(eval_history, fwrite)
    return all_results
