from transformers import AutoTokenizer
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from dataclasses import dataclass, field
from src.evaluation.metrics import DotProductMetric, Metric
from src.evaluation.eval import EvalFindClassificationThreshold, EvalClassification, EvalRanking
from src.config.dataset import DATASET_CONFIG
from src.preprocess.dataset import RJPairSimplifiedDataset, rj_pair_collate_fn
import pandas as pd
import pickle
import torch
import os
import json
import numpy as np
import faiss
import time


@dataclass
class EmbeddingNetworkTestArguments:
    model_path: str = field(
        default="model_checkpoints/dual_encoder_sft/aliyun_1e-9_decay",
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


def load_test_data(test_args: EmbeddingNetworkTestArguments):
    all_resume_data = pd.read_csv(test_args.resume_data_path)
    all_job_data = pd.read_csv(test_args.job_data_path)
    all_resume_data_dict = all_resume_data.to_dict("records")
    all_job_data_dict = all_job_data.to_dict("records")

    all_test_pairs_to_predict = []
    seen_uid = set()
    seen_jdno = set()

    test_classification_labels = pd.read_json(
        test_args.classification_data_path, lines=True
    )
    valid_classification_labels = pd.read_json(
        test_args.classification_validation_data_path, lines=True
    )
    test_ranking_resume_labels = json.load(open(test_args.rank_resume_data_path, encoding="utf-8"))
    test_ranking_job_labels = json.load(open(test_args.rank_job_data_path, encoding="utf-8"))


    for i, row in valid_classification_labels.iterrows():
        user_id = row["user_id"]
        jd_no = row["jd_no"]
        satisfied = row["satisfied"]
        if user_id in seen_uid and jd_no in seen_jdno:
            continue
        
        all_test_pairs_to_predict.append(
            {"user_id": str(user_id), "jd_no": str(jd_no), "satisfied": int(satisfied)}
        )
        seen_uid.add(user_id)
        seen_jdno.add(jd_no)
    
    for i, row in test_classification_labels.iterrows():
        user_id = row["user_id"]
        jd_no = row["jd_no"]
        satisfied = row["satisfied"]
        if user_id in seen_uid and jd_no in seen_jdno:
            continue
        
        all_test_pairs_to_predict.append(
            {"user_id": str(user_id), "jd_no": str(jd_no), "satisfied": int(satisfied)}
        )
        seen_uid.add(user_id)
        seen_jdno.add(jd_no)

    for jd_no, data in test_ranking_resume_labels.items():
        user_ids = data["user_ids"]
        satisfieds = data["satisfied"]
        for user_id, satisfied in zip(user_ids, satisfieds):
            if user_id in seen_uid and jd_no in seen_jdno:
                continue
            all_test_pairs_to_predict.append(
                {"user_id": str(user_id), "jd_no": str(jd_no), "satisfied": int(satisfied)}
            )
            seen_uid.add(user_id)
            seen_jdno.add(jd_no)

    for user_id, data in test_ranking_job_labels.items():
        jd_nos = data["jd_nos"]
        satisfieds = data["satisfied"]
        for jd_no, satisfied in zip(jd_nos, satisfieds):
            if user_id in seen_uid and jd_no in seen_jdno:
                continue
            all_test_pairs_to_predict.append(
                {"user_id": str(user_id), "jd_no": str(jd_no), "satisfied": int(satisfied)}
            )
            seen_uid.add(user_id)
            seen_jdno.add(jd_no)

    print(f"Total number of test pairs to compute: {len(all_test_pairs_to_predict)}")
    return {
        "all_resume_data_dict": all_resume_data_dict,
        "all_job_data_dict": all_job_data_dict,
        "all_test_pairs_to_embed": all_test_pairs_to_predict,
    }


def get_metric_and_representations(
    model, model_args, test_args: EmbeddingNetworkTestArguments, all_test_data: dict
) -> Metric:
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
        label_pairs=all_test_data["all_test_pairs_to_embed"],
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
        
        batched_resume_representation = output.batched_resume_representation
        batched_job_representation = output.batched_job_representation

        for offset in range(batched_resume_representation.shape[0]):
            pair = all_test_data["all_test_pairs_to_embed"][base_idx + offset]
            user_id = pair["user_id"]
            jd_no = pair["jd_no"]

            rid_to_representation[user_id] = batched_resume_representation[offset].cpu().numpy()
            jid_to_representation[jd_no] = batched_job_representation[offset].cpu().numpy()

    embedding_file_save_path = os.path.join(test_args.model_path, "test_embeddings.pkl")
    with open(embedding_file_save_path, "wb") as fwrite:
        pickle.dump(
            {
                "rid_to_representation": rid_to_representation,
                "jid_to_representation": jid_to_representation,
            },
            fwrite,
        )
    return None, rid_to_representation, jid_to_representation


def evaluate(
    metric: Metric,
    test_rid_to_representation: dict,
    test_jid_to_representation: dict,
    test_args: EmbeddingNetworkTestArguments,
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
    
    # redefine here
    metric = DotProductMetric(normalize=True)
    evaluator = EvalFindClassificationThreshold(
        metric,
        test_rid_to_representation=test_rid_to_representation,
        test_jid_to_representation=test_jid_to_representation,
        test_pairs=valid_labels_tuple,
        offline_mode=False,
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
    
    metric = DotProductMetric(normalize=True)
    evaluator = EvalClassification(
        metric,
        test_rid_to_representation=test_rid_to_representation,
        test_jid_to_representation=test_jid_to_representation,
        test_pairs=test_labels_tuple,
        threshold=best_threshold,
        offline_mode=False,
    )

    score_report, eval_history = evaluator.evaluate()
    all_results = {
        **all_results,
        **score_report
    }

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
    # redefine here
    metric = DotProductMetric(normalize=False)
    evaluator = EvalRanking(
        metric,
        test_rid_to_representation=test_rid_to_representation,
        test_jid_to_representation=test_jid_to_representation,
        test_ranking_data=test_ranking_labels,
        offline_mode=False,
    )

    score_report, eval_history = evaluator.evaluate()
    for k, v in score_report.items():
        all_results[f'rank_job_{k}'] = v

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
    metric = DotProductMetric(normalize=False)
    evaluator = EvalRanking(
        metric,
        test_rid_to_representation=test_rid_to_representation,
        test_jid_to_representation=test_jid_to_representation,
        test_ranking_data=test_ranking_labels,
        offline_mode=False,
    )

    score_report, eval_history = evaluator.evaluate()
    for k, v in score_report.items():
        all_results[f'rank_resume_{k}'] = v
    
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


def evaluate_speed(model, model_args, test_args: EmbeddingNetworkTestArguments, all_test_data: dict):
    # initialization: compute embeddings
    config = DATASET_CONFIG[test_args.dataset_type]
    max_seq_len_per_feature = config["max_seq_len_per_feature"]
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
            label_pairs=all_test_data["all_test_pairs_to_embed"][:num_to_actually_embed],
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
        to_query_resume_embeddings = []
        to_search_job_embeddings = []

        _initialization_start = time.time()
        for batch_idx, batch in tqdm(
            enumerate(test_dataloader), total=len(test_dataloader)
        ):
            with torch.no_grad():
                output = model.predict_step(batch, batch_idx)
            
            batched_resume_representation = output.batched_resume_representation
            batched_job_representation = output.batched_job_representation

            for offset in range(batched_resume_representation.shape[0]):
                to_query_resume_embeddings.append(batched_resume_representation[offset].cpu().numpy())
                to_search_job_embeddings.append(batched_job_representation[offset].cpu().numpy())
        _initialization_spent = time.time() - _initialization_start
        _initialization_spent *= (n_j_to_rank / num_to_actually_embed)
        _initialization_spent /= 2  # since we embedded both resumes and jobs
        print(f"Initialization time spent: {_initialization_spent:.2f} seconds for {n_j_to_rank=}")

        to_query_resume_embeddings_duped = []
        to_search_job_embeddings_duped = []
        for _ in range(n_j_to_rank // num_to_actually_embed):
            to_query_resume_embeddings_duped.extend(to_query_resume_embeddings)
            to_search_job_embeddings_duped.extend(to_search_job_embeddings)

        to_query_resume_embeddings_vec = np.vstack(to_query_resume_embeddings)
        to_search_job_embeddings_vec = np.vstack(to_search_job_embeddings)

        ### create the faiss embeddings
        index = faiss.IndexFlatIP(to_search_job_embeddings_vec.shape[1])
        index.add(to_search_job_embeddings_vec)
        
        print('added')

        # search once and time it
        _search_start = time.time()
        top_k = 10
        _, _ = index.search(to_query_resume_embeddings_vec[:1], top_k)  # find top 10
        _search_spent = time.time() - _search_start
        print(f"Search time spent: {_search_spent:.2f} seconds for {n_j_to_rank=}")

        ## store stats
        stats_dict[f'rank{n_j_to_rank}_init'] = _initialization_spent
        stats_dict[f'rank{n_j_to_rank}_search'] = _search_spent
    return stats_dict