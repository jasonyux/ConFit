from src.utils.test_embedding_networks import (
    EmbeddingNetworkTestArguments,
    evaluate,
)
import os
import json
import jsonlines
import argparse


def configure_test_data(args):
    if args.dset == 'IntelliPro':
        test_args = EmbeddingNetworkTestArguments(
            model_path=args.embedding_folder,
            resume_data_path='dataset/IntelliPro/all_resume.csv',
            job_data_path='dataset/IntelliPro/all_jd.csv',
            classification_validation_data_path='dataset/IntelliPro/valid_classification_data.jsonl',
            classification_data_path='dataset/IntelliPro/test_classification_data.jsonl',
            rank_resume_data_path='dataset/IntelliPro/rank_resume.json',
            rank_job_data_path='dataset/IntelliPro/rank_job.json',
            dataset_type='IntelliPro',
            query_prefix='',
            batch_size=16,
            seed=42,
        )
    else:
        test_args = EmbeddingNetworkTestArguments(
            model_path=args.embedding_folder,
            resume_data_path='dataset/AliTianChi/all_resume.csv',
            job_data_path='dataset/AliTianChi/all_jd.csv',
            classification_validation_data_path='dataset/AliTianChi/valid_classification_data.jsonl',
            classification_data_path='dataset/AliTianChi/test_classification_data.jsonl',
            rank_resume_data_path='dataset/AliTianChi/rank_resume.json',
            rank_job_data_path='dataset/AliTianChi/rank_job.json',
            dataset_type='AliTianChi',
            query_prefix='',
            batch_size=16,
            seed=42,
        )
    return test_args


def load_embeddings(args: argparse.Namespace):
    resume_embedding_path = os.path.join(args.embedding_folder, 'all_resume_test_embeddings.jsonl')
    with jsonlines.open(resume_embedding_path, 'r') as reader:
        every_resume_to_test = list(reader)
        print('loaded', len(every_resume_to_test))

    job_embedding_path = os.path.join(args.embedding_folder, 'all_job_test_embeddings.jsonl')
    with jsonlines.open(job_embedding_path, 'r') as reader:
        every_job_to_test = list(reader)
        print('loaded', len(every_job_to_test))

    every_resume_to_test_dict = {}
    for d in every_resume_to_test:
        every_resume_to_test_dict[d['rid']] = d['embedding']

    every_job_to_test_dict = {}
    for d in every_job_to_test:
        every_job_to_test_dict[d['jid']] = d['embedding']
    return every_resume_to_test_dict, every_job_to_test_dict


def eval_performance(
    every_resume_to_test_dict,
    every_job_to_test_dict,
    test_args: EmbeddingNetworkTestArguments
):
    eval_results = evaluate(
        metric=None,
        test_rid_to_representation=every_resume_to_test_dict,
        test_jid_to_representation=every_job_to_test_dict,
        test_args=test_args,
    )
    return eval_results



def main(args: argparse.Namespace):
    resume_embeddings, job_embeddings = load_embeddings(args)
    test_args = configure_test_data(args)
    embedding_perf = eval_performance(
        resume_embeddings,
        job_embeddings,
        test_args
    )

    print(json.dumps(embedding_perf, indent=4))
    return

if __name__ == '__main__':
    # for examples, see
    # model_checkpoints/openai/aliyun/all_resume_test_embeddings.jsonl
    # model_checkpoints/bert-base-chinese/aliyun/all_resume_test_embeddings.jsonl
    # model_checkpoints/openai/IntelliPro/all_resume_test_embeddings.jsonl
    # model_checkpoints/bert-base-multilingual-cased/IntelliPro/all_resume_test_embeddings.jsonl
    # model_checkpoints/multilingual-e5-large/aliyun/all_resume_test_embeddings.jsonl
    # model_checkpoints/multilingual-e5-large/IntelliPro/all_resume_test_embeddings.jsonl

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dset', type=str, default='IntelliPro',
        choices=['IntelliPro', 'aliyun']
    )
    parser.add_argument(
        '--embedding_folder', type=str,
        default='model_checkpoints/openai/IntelliPro',
    )
    args = parser.parse_args()
    main(args)