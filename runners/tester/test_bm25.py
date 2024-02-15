from src.evaluation.metrics import PrecomputedMetric
from src.evaluation.eval import EvalClassification, EvalRanking, EvalFindClassificationThreshold
from src.model.tfidf import tfidf_train, tfidf_predict
from src.model.bow import bow_train, bow_predict
from src.preprocess.word_tokenize import (
    proprietary_resume_to_words_simple, proprietary_jd_to_words_simple,
    ali_resume_to_words_simple, ali_jd_to_words_simple,
)
from rank_bm25 import BM25L
from typing import List, Set, Callable, Dict
from tqdm.auto import tqdm
from multiprocessing import Pool
import json
import pandas as pd
import numpy as np
import argparse


def mp_proprietary_resume_to_words_simple(resume_data: dict):
    """used by MP to make it faster
    """
    user_id = resume_data['user_id']
    output_words = proprietary_resume_to_words_simple(resume_data)
    return user_id, output_words

def mp_proprietary_jd_to_words_simple(job_data: dict):
    """used by MP to make it faster
    """
    jd_no = job_data['jd_no']
    output_words = proprietary_jd_to_words_simple(job_data)
    return jd_no, output_words


def mp_ali_resume_to_words_simple(resume_data: dict):
    """used by MP to make it faster
    """
    user_id = resume_data['user_id']
    output_words = ali_resume_to_words_simple(resume_data)
    return user_id, output_words


def mp_ali_jd_to_words_simple(job_data: dict):
    """used by MP to make it faster
    """
    jd_no = job_data['jd_no']
    output_words = ali_jd_to_words_simple(job_data)
    return jd_no, output_words


def tokenize_job(
    all_job_data: pd.DataFrame,
    modify_one_job: Callable[[dict], dict],
    tokenize_one_job: Callable[[dict], List[str]],
    trainable_jids: set
):
    # job tokenizatino is the same as before
    job_id_to_representations = {}
    
    job_to_display = []
    all_job_dicts = all_job_data.to_dict('records')
    all_job_dicts = [modify_one_job(job_dict) for job_dict in all_job_dicts]
    
    pbar = tqdm(total=len(all_job_data), desc='tokenizing job')
    with Pool(processes=8) as pool:
        for jid, job_words in pool.imap_unordered(tokenize_one_job, all_job_dicts):
            pbar.update(1)
            assert "aug_" not in jid, f"augmented jobs should not be used here: {jid}"

            if jid in trainable_jids:
                continue

            # all we need is the tokenized corpus
            job_id_to_representations[jid] = job_words

            if len(job_to_display) < 2:
                job_to_display.append(job_words)
    pbar.close()

    print('job_to_display')
    print(job_to_display)
    return job_id_to_representations


def tokenize_resume(
    all_resume_data: pd.DataFrame,
    modify_one_resume: Callable[[dict], dict],
    tokenize_resume: Callable[[dict], List[str]],
    trainable_rids: set
):
    resume_id_to_representations = {}

    resume_to_display = []
    all_resume_dicts = all_resume_data.to_dict('records')
    all_resume_dicts = [modify_one_resume(resume_dict) for resume_dict in all_resume_dicts]

    pbar = tqdm(total=len(all_resume_data), desc='tokenizing resume')
    with Pool(processes=8) as pool:
        for rid, resume_words in pool.imap_unordered(tokenize_resume, all_resume_dicts):
            pbar.update(1)
            assert "aug_" not in rid, f"augmented resumes should not be used here: {rid}"
            if rid in trainable_rids:
                continue

            resume_id_to_representations[rid] = resume_words

            if len(resume_to_display) < 2:
                resume_to_display.append(resume_words)
    pbar.close()
    
    print('resume_to_display')
    print(resume_to_display)
    return resume_id_to_representations


def evaluate_ranking(
    ranking_test_labels: Dict[str, dict],
    ranking_metric: PrecomputedMetric,
):
    
    evaluator = EvalRanking(
        ranking_metric,
        test_rid_to_representation={},
        test_jid_to_representation={},
        test_ranking_data=ranking_test_labels,
        offline_mode=True
    )

    rank_score_report, _ = evaluator.evaluate()
    return rank_score_report


def get_BM25_metric(
    rank_job_test_labels: Dict[str, dict],
    rank_resume_test_labels: Dict[str, dict],
    resume_id_to_representations: Dict[str, np.ndarray],
    job_id_to_representations: Dict[str, np.ndarray],
):
    all_pairs_scored = {}
    ## prepare all documents
    all_resume_docs = []
    all_rid_to_dids = {}
    all_job_docs = []
    all_jid_to_dids = {}
    for rid, data in rank_job_test_labels.items():
        if rid not in all_rid_to_dids:
            all_resume_docs.append(resume_id_to_representations[rid])
            all_rid_to_dids[rid] = len(all_resume_docs) - 1
        jids = data['jd_nos']
        for jid in jids:
            if jid not in all_jid_to_dids:
                all_job_docs.append(job_id_to_representations[jid])
                all_jid_to_dids[jid] = len(all_job_docs) - 1
    for jid, data in rank_resume_test_labels.items():
        if jid not in all_jid_to_dids:
            all_job_docs.append(job_id_to_representations[jid])
            all_jid_to_dids[jid] = len(all_job_docs) - 1
        rids = data['user_ids']
        for rid in rids:
            if rid not in all_rid_to_dids:
                all_resume_docs.append(resume_id_to_representations[rid])
                all_rid_to_dids[rid] = len(all_resume_docs) - 1

    bm25_rank_resume = BM25L(all_resume_docs)
    for jid, data in rank_resume_test_labels.items():
        query = job_id_to_representations[jid]
        rids = data['user_ids']
        # get the dids
        dids = [all_rid_to_dids[rid] for rid in rids]
        scores = bm25_rank_resume.get_batch_scores(query, dids)
        for s, rid in zip(scores, rids):
            all_pairs_scored[(rid, jid)] = s
    bm25_rank_job = BM25L(all_job_docs)
    for rid, data in rank_job_test_labels.items():
        query = resume_id_to_representations[rid]
        jids = data['jd_nos']
        # get the dids
        dids = [all_jid_to_dids[jid] for jid in jids]
        scores = bm25_rank_job.get_batch_scores(query, dids)
        for s, jid in zip(scores, jids):
            all_pairs_scored[(rid, jid)] = s

    metric = PrecomputedMetric(all_pairs_scored)
    return metric

def eval_gold(
    all_job_data: pd.DataFrame,
    all_resume_data: pd.DataFrame,
    all_trainable_jids: Set[str],
    all_trainable_rids: Set[str],
    tokenize_one_job: Callable[[dict], List[str]],
    tokenize_one_resume: Callable[[dict], List[str]],
    rank_job_test_labels: Dict[str, dict],
    rank_resume_test_labels: Dict[str, dict],
):
    job_id_to_representations = tokenize_job(
        all_job_data,
        modify_one_job = lambda x: x,  # no modification
        tokenize_one_job = tokenize_one_job,
        trainable_jids = all_trainable_jids
    )

    resume_id_to_representations = tokenize_resume(
        all_resume_data,
        modify_one_resume = lambda x: x,
        tokenize_resume = tokenize_one_resume,
        trainable_rids = all_trainable_rids
    )
    
    print('resume_id_to_representations.size', len(resume_id_to_representations))
    for _, rep in resume_id_to_representations.items():
        print('len(rep)', len(rep))
        break
    
    ### prepare the training data/validation splits and train DT
    metric = get_BM25_metric(
        rank_job_test_labels,
        rank_resume_test_labels,
        resume_id_to_representations,
        job_id_to_representations,
    )

    # skipped classification for BM25 as it is wierd
    # BM25 is for ranking asssymetric things, but classification task is symmetric

    # ranking tests
    rank_job_score_report = evaluate_ranking(
        rank_job_test_labels,
        metric
    )
    rank_resume_score_report = evaluate_ranking(
        rank_resume_test_labels,
        metric
    )

    all_score_report = {
        'rank_job': rank_job_score_report.copy(),
        'rank_resume': rank_resume_score_report.copy()
    }
    return all_score_report


def load_data(
    args: argparse.Namespace,
):
    if args.dset == 'IntelliPro':
        print('loading IntelliPro data')
        all_resume_data = pd.read_csv('dataset/IntelliPro/all_resume.csv')
        all_resume_data['user_id'] = all_resume_data['user_id'].astype(str)
        all_resume_data.index = all_resume_data['user_id'].values

        all_job_data = pd.read_csv('dataset/IntelliPro/all_jd.csv')
        all_job_data['jd_no'] = all_job_data['jd_no'].astype(str)
        all_job_data.index = all_job_data['jd_no'].values

        # we allow for augmented data by adding the line "i_no_prefix in trainable_rids"
        trainable_jids = set(pd.read_csv('dataset/IntelliPro/trainable_jd.csv')['jd_no'].astype(str).values)
        trainable_rids = set(pd.read_csv('dataset/IntelliPro/trainable_resume.csv')['user_id'].astype(str).values)

        rank_job_test_labels = json.load(open('dataset/IntelliPro/rank_job.json', encoding='utf-8'))
        rank_resume_test_labels = json.load(open('dataset/IntelliPro/rank_resume.json', encoding='utf-8'))
    else:
        print('loading aliyun data')
        all_resume_data = pd.read_csv('dataset/AliTianChi/all_resume.csv')
        all_resume_data['user_id'] = all_resume_data['user_id'].astype(str)
        all_resume_data.index = all_resume_data['user_id'].values

        all_job_data = pd.read_csv('dataset/AliTianChi/all_jd.csv')
        all_job_data['jd_no'] = all_job_data['jd_no'].astype(str)
        all_job_data.index = all_job_data['jd_no'].values

        trainable_jids = set(pd.read_csv('dataset/AliTianChi/trainable_jd.csv')['jd_no'].astype(str).values)
        trainable_rids = set(pd.read_csv('dataset/AliTianChi/trainable_resume.csv')['user_id'].astype(str).values)

        rank_job_test_labels = json.load(open('dataset/AliTianChi/rank_job.json', encoding='utf-8'))
        rank_resume_test_labels = json.load(open('dataset/AliTianChi/rank_resume.json', encoding='utf-8'))
    return {
        'all_resume_data': all_resume_data,
        'all_job_data': all_job_data,
        'trainable_jids': trainable_jids,
        'trainable_rids': trainable_rids,
        'rank_job_test_labels': rank_job_test_labels,
        'rank_resume_test_labels': rank_resume_test_labels,
    }


def main(args: argparse.Namespace):
    all_data = load_data(args)
    xgboost_perf = eval_gold(
        all_data['all_job_data'],
        all_data['all_resume_data'],
        all_data['trainable_jids'],
        all_data['trainable_rids'],
        mp_proprietary_jd_to_words_simple if args.dset == 'IntelliPro' else mp_ali_jd_to_words_simple,
        mp_proprietary_resume_to_words_simple if args.dset == 'IntelliPro' else mp_ali_resume_to_words_simple,
        all_data['rank_job_test_labels'],
        all_data['rank_resume_test_labels'],
    )

    print(json.dumps(xgboost_perf, indent=4))
    return

if __name__ == '__main__':
    # example
    # CUDA_VISIBLE_DEVICES=0 python runners/tester/test_bm25.py  \
    # --dset IntelliPro --encoder tfidf
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dset', type=str, default='IntelliPro',
        choices=['IntelliPro', 'aliyun']
    )
    args = parser.parse_args()
    main(args)