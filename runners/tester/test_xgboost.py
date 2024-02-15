from src.evaluation.metrics import DecisionTreeMetric, DecisionTreeRankerMetric
from src.evaluation.eval import EvalClassification, EvalRanking
from src.model.tfidf import tfidf_train, tfidf_predict
from src.model.bow import bow_train, bow_predict
from src.preprocess.word_tokenize import (
    proprietary_resume_to_words, proprietary_jd_to_words,
    proprietary_resume_to_words_simple, proprietary_jd_to_words_simple,
    ali_resume_to_words, ali_jd_to_words,
    ali_resume_to_words_simple, ali_jd_to_words_simple,
)
from typing import List, Set, Callable, Dict
from functools import partial
from tqdm.auto import tqdm
import json
import pandas as pd
import numpy as np
import argparse


def tokenize_job(
    all_job_data: pd.DataFrame,
    modify_one_job: Callable[[dict], dict],
    tokenize_one_job: Callable[[dict], List[str]],
    embed_training_jobs: Callable[[List], np.ndarray],
    embed_test_jobs: Callable[[List], np.ndarray],
    max_features: int,
    trainable_jids: set
):
    # job tokenizatino is the same as before
    job_id_to_representations = {}
    _trainable_job_tokenized = []
    _trainable_job_indices = []
    _to_predict_job_tokenized = []
    _to_predict_job_indices = []

    job_to_display = []
    pbar = tqdm(total=len(all_job_data), desc='tokenizing job')
    for i, job in all_job_data.iterrows():
        job_dict = job.to_dict().copy()
        job_dict = modify_one_job(job_dict)
        job_words = tokenize_one_job(job_dict)
        if i in trainable_jids:
            _trainable_job_tokenized.append(job_words)
            _trainable_job_indices.append(i)
        else:
            _to_predict_job_tokenized.append(job_words)
            _to_predict_job_indices.append(i)

        if len(job_to_display) < 2:
            job_to_display.append(job_words)
        pbar.update(1)
    pbar.close()

    trained_job_tfidf, vectorizer = embed_training_jobs(_trainable_job_tokenized, max_features=max_features)
    trained_job_tfidf = trained_job_tfidf.toarray()
    for i, j in zip(_trainable_job_indices, trained_job_tfidf):
        job_id_to_representations[i] = j
    
    predicted_job_tfidf = embed_test_jobs(_to_predict_job_tokenized, vectorizer)
    predicted_job_tfidf = predicted_job_tfidf.toarray()
    for i, j in zip(_to_predict_job_indices, predicted_job_tfidf):
        job_id_to_representations[i] = j

    print('job_to_display')
    print(job_to_display)
    print('vectorizer features')
    print(vectorizer.get_feature_names_out())
    return job_id_to_representations


def tokenize_resume(
    all_resume_data: pd.DataFrame,
    modify_one_resume: Callable[[dict], dict],
    tokenize_resume: Callable[[dict], List[str]],
    embed_training_resumes: Callable[[List], np.ndarray],
    embed_test_resumes: Callable[[List], np.ndarray],
    max_features,
    trainable_rids: set
):
    resume_id_to_representations = {}

    ##### Prepare training data in STRING
    # remove one resume key at a time
    _trainable_resume_tokenized = []
    _trainable_resume_indices = []
    _to_predict_resume_tokenized = []
    _to_predict_resume_indices = []

    resume_to_display = []
    pbar = tqdm(total=len(all_resume_data), desc='tokenizing resume')
    for i, resume in all_resume_data.iterrows():
        resume_dict = resume.to_dict().copy()
        resume_dict = modify_one_resume(resume_dict)

        resume_words = tokenize_resume(resume_dict)
        if i in trainable_rids:
            _trainable_resume_tokenized.append(resume_words)
            _trainable_resume_indices.append(i)
        else:
            _to_predict_resume_tokenized.append(resume_words)
            _to_predict_resume_indices.append(i)

        if len(resume_to_display) < 2:
            resume_to_display.append(resume_words)
        pbar.update(1)
    pbar.close()
    
    trained_resume_tfidf, vectorizer = embed_training_resumes(_trainable_resume_tokenized, max_features=max_features)
    trained_resume_tfidf = trained_resume_tfidf.toarray()
    for i, j in zip(_trainable_resume_indices, trained_resume_tfidf):
        resume_id_to_representations[i] = j
    
    predicted_resume_tfidf = embed_test_resumes(_to_predict_resume_tokenized, vectorizer)
    predicted_resume_tfidf = predicted_resume_tfidf.toarray()
    for i, j in zip(_to_predict_resume_indices, predicted_resume_tfidf):
        resume_id_to_representations[i] = j
    
    print('resume_to_display')
    print(resume_to_display)
    print('vectorizer features')
    print(vectorizer.get_feature_names_out())
    return resume_id_to_representations


def reorder_ranking_data(resume_repr, job_repr, labels, group_ids):
    ## XGBoost need the data to be sorted by group id
    sorted_idx = np.argsort(group_ids)
    resume_repr = resume_repr[sorted_idx, :]
    job_repr = job_repr[sorted_idx, :]
    labels = labels[sorted_idx]
    group_ids = group_ids[sorted_idx]
    return resume_repr, job_repr, labels, group_ids


def train_DT_metric(
    training_labels: pd.DataFrame,
    validation_labels: pd.DataFrame,
    resume_id_to_representations: dict,
    job_id_to_representations: dict,
):
    train_qids = 0
    train_resume_representations = []
    train_job_representations = []
    train_labels = []
    train_rank_resume_group_ids = []  # given a job, rank resume. So group id is job id
    __train_rank_resume_group_id_mapping = {}
    train_rank_job_group_ids = []
    __train_rank_job_group_id_mapping = {}

    for _, row in training_labels.iterrows():
        user_id = row['user_id']
        jd_no = row['jd_no']
        train_resume_representations.append(resume_id_to_representations[user_id])
        train_job_representations.append(job_id_to_representations[jd_no])
        train_labels.append(int(row['satisfied']))

        if jd_no not in __train_rank_resume_group_id_mapping:
            __train_rank_resume_group_id_mapping[jd_no] = train_qids
            train_qids += 1
        train_rank_resume_group_ids.append(__train_rank_resume_group_id_mapping[jd_no])

        if user_id not in __train_rank_job_group_id_mapping:
            __train_rank_job_group_id_mapping[user_id] = train_qids
            train_qids += 1
        train_rank_job_group_ids.append(__train_rank_job_group_id_mapping[user_id])
    train_resume_representations = np.array(train_resume_representations)
    train_job_representations = np.array(train_job_representations)

    valid_qids = 0
    valid_resume_representations = []
    valid_job_representations = []
    valid_labels = []
    valid_rank_resume_group_ids = []  # given a job, rank resume. So group id is job id
    __valid_rank_resume_group_id_mapping = {}
    valid_rank_job_group_ids = []
    __valid_rank_job_group_id_mapping = {}

    for _, row in validation_labels.iterrows():
        user_id = row['user_id']
        jd_no = row['jd_no']
        valid_resume_representations.append(resume_id_to_representations[user_id])
        valid_job_representations.append(job_id_to_representations[jd_no])
        valid_labels.append(int(row['satisfied']))

        if jd_no not in __valid_rank_resume_group_id_mapping:
            __valid_rank_resume_group_id_mapping[jd_no] = valid_qids
            valid_qids += 1
        valid_rank_resume_group_ids.append(__valid_rank_resume_group_id_mapping[jd_no])

        if user_id not in __valid_rank_job_group_id_mapping:
            __valid_rank_job_group_id_mapping[user_id] = valid_qids
            valid_qids += 1
        valid_rank_job_group_ids.append(__valid_rank_job_group_id_mapping[user_id])
    valid_resume_representations = np.array(valid_resume_representations)
    valid_job_representations = np.array(valid_job_representations)


    #### Train and Predict
    print('train_resume_representations')
    print(f"==>> train_resume_representations.shape: {train_resume_representations.shape}")
    print(train_resume_representations.sum(axis=1))
    print('percent element that has non-zero values')
    print(1.0 - ( np.count_nonzero(train_resume_representations) / float(train_resume_representations.size) ))  # sparsity
    print('train_job_representations')
    print(f"==>> train_job_representations.shape: {train_job_representations.shape}")
    print(train_job_representations.sum(axis=1))
    
    metric = DecisionTreeMetric(
        train_resume_representations,
        train_job_representations,
        train_labels,
        valid_resume_representations,
        valid_job_representations,
        valid_labels,
        merge_op = 'concat',
        do_sweep = True,
        model_save_path=None,
    )

    ranking_hparam = metric.training_hparams.copy()
    ranking_hparam.pop('early_stopping_rounds')
    ranking_hparam['objective'] = 'rank:map'

    train_resume_repr, train_job_repr, train_y, train_gids = reorder_ranking_data(
        train_resume_representations,
        train_job_representations,
        np.array(train_labels),
        np.array(train_rank_job_group_ids)
    )

    valid_resume_repr, valid_job_repr, valid_y, valid_gids = reorder_ranking_data(
        valid_resume_representations,
        valid_job_representations,
        np.array(valid_labels),
        np.array(valid_rank_job_group_ids)
    )

    rank_job_metric = DecisionTreeRankerMetric(
        train_resume_repr,
        train_job_repr,
        train_y,
        train_gids,
        valid_resume_repr,
        valid_job_repr,
        valid_y,
        valid_gids,
        merge_op = 'concat',
        do_sweep = False,  # we don't reallly have a valid set for ranking
        model_save_path=None,
        training_hparams=ranking_hparam
    )

    train_resume_repr, train_job_repr, train_y, train_gids = reorder_ranking_data(
        train_resume_representations,
        train_job_representations,
        np.array(train_labels),
        np.array(train_rank_resume_group_ids)
    )

    valid_resume_repr, valid_job_repr, valid_y, valid_gids = reorder_ranking_data(
        valid_resume_representations,
        valid_job_representations,
        np.array(valid_labels),
        np.array(valid_rank_resume_group_ids)
    )

    rank_resume_metric = DecisionTreeRankerMetric(
        train_resume_repr,
        train_job_repr,
        train_y,
        train_gids,
        valid_resume_repr,
        valid_job_repr,
        valid_y,
        valid_gids,
        merge_op = 'concat',
        do_sweep = False,  # we don't reallly have a valid set for ranking
        model_save_path=None,
        training_hparams=ranking_hparam
    )
    return metric, rank_job_metric, rank_resume_metric


def evaluate_classification(
    classification_test_labels: pd.DataFrame,
    resume_id_to_representations: dict,
    job_id_to_representations: dict,
    metric: DecisionTreeMetric,
):

    test_rid_to_representation = {}
    test_jid_to_representation = {}
    test_pairs = []

    for i, row in classification_test_labels.iterrows():
        test_pairs.append([row['user_id'], row['jd_no'], int(row['satisfied'])])
        user_id = row['user_id']
        jd_no = row['jd_no']
        test_rid_to_representation[user_id] = resume_id_to_representations[user_id]
        test_jid_to_representation[jd_no] = job_id_to_representations[jd_no]

    evaluator = EvalClassification(
        metric,
        test_rid_to_representation,
        test_jid_to_representation,
        test_pairs
    )

    classfication_score_report, _ = evaluator.evaluate()
    return classfication_score_report


def evaluate_ranking(
    ranking_test_labels: Dict[str, dict],
    resume_id_to_representations: dict,
    job_id_to_representations: dict,
    ranking_metric: DecisionTreeRankerMetric,
):
    test_rid_to_representation = {}
    test_jid_to_representation = {}
    # check dataset type
    dset_type = 'rank_job'
    for user_id, data in ranking_test_labels.items():
        if 'jd_nos' not in data:
            dset_type = 'rank_resume'
        break

    if dset_type == 'rank_job':
        for user_id, data in ranking_test_labels.items():
            test_rid_to_representation[user_id] = resume_id_to_representations[user_id]
            for jd_no in data['jd_nos']:
                test_jid_to_representation[jd_no] = job_id_to_representations[jd_no]
    else:
        for jd_no, data in ranking_test_labels.items():
            test_jid_to_representation[jd_no] = job_id_to_representations[jd_no]
            for user_id in data['user_ids']:
                test_rid_to_representation[user_id] = resume_id_to_representations[user_id]
    
    evaluator = EvalRanking(
        ranking_metric,
        test_rid_to_representation,
        test_jid_to_representation,
        test_ranking_data=ranking_test_labels
    )

    rank_score_report, _ = evaluator.evaluate()
    return rank_score_report

def eval_gold(
    all_job_data: pd.DataFrame,
    all_resume_data: pd.DataFrame,
    all_trainable_jids: Set[str],
    all_trainable_rids: Set[str],
    tokenize_one_job: Callable[[dict], List[str]],
    tokenize_one_resume: Callable[[dict], List[str]],
    embed_training_texts: Callable[[List], np.ndarray],
    embed_testing_texts: Callable[[List], np.ndarray],
    training_labels: pd.DataFrame,
    validation_labels: pd.DataFrame,
    classification_test_labels: pd.DataFrame,
    rank_job_test_labels: Dict[str, dict],
    rank_resume_test_labels: Dict[str, dict],
):
    max_features = 768

    job_id_to_representations = tokenize_job(
        all_job_data,
        modify_one_job = lambda x: x,  # no modification
        tokenize_one_job = tokenize_one_job,
        embed_training_jobs = embed_training_texts,
        embed_test_jobs = embed_testing_texts,
        max_features = max_features,
        trainable_jids = all_trainable_jids
    )

    resume_id_to_representations = tokenize_resume(
        all_resume_data,
        modify_one_resume = lambda x: x,
        tokenize_resume = tokenize_one_resume,
        embed_training_resumes = embed_training_texts,
        embed_test_resumes = embed_testing_texts,
        max_features = max_features,
        trainable_rids = all_trainable_rids
    )
    
    print('resume_id_to_representations.size', len(resume_id_to_representations))
    for _, rep in resume_id_to_representations.items():
        print(rep.shape)
        break
    
    ### prepare the training data/validation splits and train DT
    metric, rank_job_metric, rank_resume_metric = train_DT_metric(
        training_labels,
        validation_labels,
        resume_id_to_representations,
        job_id_to_representations,
    )

    # get scores
    classfication_score_report = evaluate_classification(
        classification_test_labels,
        resume_id_to_representations,
        job_id_to_representations,
        metric
    )

    # ranking tests
    best_rank_job_report = {}
    rank_job_score_report = evaluate_ranking(
        rank_job_test_labels,
        resume_id_to_representations,
        job_id_to_representations,
        rank_job_metric
    )
    rank_job_score_report_dt = evaluate_ranking(
        rank_job_test_labels,
        resume_id_to_representations,
        job_id_to_representations,
        metric
    )
    # report the better one
    for k, v in rank_job_score_report.items():
        if v > rank_job_score_report_dt[k]:
            best_rank_job_report[k] = v
        else:
            best_rank_job_report[k] = rank_job_score_report_dt[k]

    # rank resume
    best_rank_resume_report = {}
    rank_resume_score_report = evaluate_ranking(
        rank_resume_test_labels,
        resume_id_to_representations,
        job_id_to_representations,
        rank_resume_metric
    )
    rank_resume_score_report_dt = evaluate_ranking(
        rank_resume_test_labels,
        resume_id_to_representations,
        job_id_to_representations,
        metric
    )
    # report the better one
    for k, v in rank_resume_score_report.items():
        if v > rank_resume_score_report_dt[k]:
            best_rank_resume_report[k] = v
        else:
            best_rank_resume_report[k] = rank_resume_score_report_dt[k]

    all_score_report = {
        'classification': classfication_score_report.copy(),
        'rank_job': best_rank_job_report.copy(),
        'rank_resume': best_rank_resume_report.copy()
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

        trainable_jids = set(pd.read_csv('dataset/IntelliPro/trainable_jd.csv')['jd_no'].astype(str).values)
        trainable_rids = set(pd.read_csv('dataset/IntelliPro/trainable_resume.csv')['user_id'].astype(str).values)

        training_labels = pd.read_json('dataset/IntelliPro/train_labeled_data.jsonl', lines=True)
        training_labels['user_id'] = training_labels['user_id'].astype(str)
        training_labels['jd_no'] = training_labels['jd_no'].astype(str)
        validation_labels = pd.read_json('dataset/IntelliPro/valid_classification_data.jsonl', lines=True)
        validation_labels['user_id'] = validation_labels['user_id'].astype(str)
        validation_labels['jd_no'] = validation_labels['jd_no'].astype(str)
        classification_test_labels = pd.read_json('dataset/IntelliPro/test_classification_data.jsonl', lines=True)
        classification_test_labels['user_id'] = classification_test_labels['user_id'].astype(str)
        classification_test_labels['jd_no'] = classification_test_labels['jd_no'].astype(str)
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

        training_labels = pd.read_json('dataset/AliTianChi/train_labeled_data.jsonl', lines=True)
        training_labels['user_id'] = training_labels['user_id'].astype(str)
        training_labels['jd_no'] = training_labels['jd_no'].astype(str)
        validation_labels = pd.read_json('dataset/AliTianChi/valid_classification_data.jsonl', lines=True)
        validation_labels['user_id'] = validation_labels['user_id'].astype(str)
        validation_labels['jd_no'] = validation_labels['jd_no'].astype(str)
        classification_test_labels = pd.read_json('dataset/AliTianChi/test_classification_data.jsonl', lines=True)
        classification_test_labels['user_id'] = classification_test_labels['user_id'].astype(str)
        classification_test_labels['jd_no'] = classification_test_labels['jd_no'].astype(str)
        rank_job_test_labels = json.load(open('dataset/AliTianChi/rank_job.json', encoding='utf-8'))
        rank_resume_test_labels = json.load(open('dataset/AliTianChi/rank_resume.json', encoding='utf-8'))

    ### sanity check data
    ### check if train ids appear in valid or test sets
    for i, row in validation_labels.iterrows():
        rid = row['user_id']
        jid = row['jd_no']
        assert rid not in trainable_rids
        assert jid not in trainable_jids

    for i, row in classification_test_labels.iterrows():
        rid = row['user_id']
        jid = row['jd_no']
        assert rid not in trainable_rids
        assert jid not in trainable_jids

    for rid, data in rank_job_test_labels.items():
        assert rid not in trainable_rids
        jids = data['jd_nos']
        for jid in jids:
            assert jid not in trainable_jids

    for jid, data in rank_resume_test_labels.items():
        assert jid not in trainable_jids
        rids = data['user_ids']
        for rid in rids:
            assert rid not in trainable_rids

    # print
    print('loaded training data', len(training_labels))
    print('loaded validation data', len(validation_labels))
    return {
        'all_resume_data': all_resume_data,
        'all_job_data': all_job_data,
        'trainable_jids': trainable_jids,
        'trainable_rids': trainable_rids,
        'training_labels': training_labels,
        'validation_labels': validation_labels,
        'classification_test_labels': classification_test_labels,
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
        proprietary_jd_to_words_simple if args.dset == 'IntelliPro' else ali_jd_to_words_simple,
        proprietary_resume_to_words_simple if args.dset == 'IntelliPro' else ali_resume_to_words_simple,
        tfidf_train if args.encoder == 'tfidf' else bow_train,
        tfidf_predict if args.encoder == 'tfidf' else bow_predict,
        all_data['training_labels'],
        all_data['validation_labels'],
        all_data['classification_test_labels'],
        all_data['rank_job_test_labels'],
        all_data['rank_resume_test_labels'],
    )

    print(json.dumps(xgboost_perf, indent=4))
    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dset', type=str, default='IntelliPro',
        choices=['IntelliPro', 'aliyun']
    )
    parser.add_argument(
        '--encoder', type=str, default='bow',
        choices=['bow', 'tfidf']
    )
    args = parser.parse_args()
    main(args)