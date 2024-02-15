from typing import List, Dict, Tuple, Callable, Any
from torch.utils.data import Dataset
from tqdm.auto import tqdm
from transformers import BatchEncoding
from transformers.data.data_collator import default_data_collator
from collections import OrderedDict
import torch
import random
import pandas as pd


class RJPairDataset(Dataset):
    """
    encode data to the format of
    a dict resume/job would become: {
        "desired_city": {
            "encoded_key_name": {},  # BatchEncoding of the string "desired_city". Shape token_id=1 * seq_length, token_type_ids=1 * seq_length, etc.
            "encoded_values": {},  # BatchEncoding of the value of "desired_city". Shape token_id=1 * seq_length, token_type_ids=1 * seq_length, etc.
        },
        ...
    }
    and additionally return
    resume_taxon_encoding: BatchEncoding of the string "resume", Shape token_id=1 * seq_length, etc.
    job_taxon_encoding: BatchEncoding of the string "job description", Shape token_id=1 * seq_length, etc.
    """

    def __init__(
        self,
        tokenizer,
        max_key_seq_length: int,
        max_seq_length_per_key: Dict[str, int],
        resume_key_names: List[str],
        job_key_names: List[str],
        tokenizer_args: Dict[str, Any],
        all_resume_dict: List[dict],
        all_job_dict: List[dict],
        label_pairs: List[Tuple[str, str, int]],
        resume_taxon_token: str = "",
        job_taxon_token: str = "",
        query_prefix: str = "",
    ):
        print("Using query_prefix:", query_prefix)

        self.tokenizer = tokenizer
        self.query_prefix = query_prefix  # used by special encoders such as e5
        self.max_key_seq_length = max_key_seq_length  # tyipcally small, like 16, since the key names are just 2-3 words
        self.max_seq_length_per_key = (
            max_seq_length_per_key  # will override the max_seq_length in tokenizer_args
        )
        self.resume_key_names = resume_key_names  # ensure the order of dict key is constant
        self.job_key_names = job_key_names
        self.tokenizer_args = tokenizer_args

        self.data = self.contruct_labeled_pairs(
            all_resume_dict, all_job_dict, label_pairs
        )
        self.encoded_data = self.encode_data(
            self.data, resume_taxon_token, job_taxon_token
        )
        return

    def contruct_labeled_pairs(
        self,
        all_resume_dict: List[dict],
        all_job_dict: List[dict],
        label_pairs: List[dict],
    ):
        """
        label_pairs: [{'user_id': "xxx", 'jd_no': "xxx", 'satisifed': 0}, ...]
        output: [(resume_dict, job_dict, label), ...]
        """
        uid_to_resume = {}
        jd_no_to_job = {}

        for resume in all_resume_dict:
            resume_ = resume.copy()
            uid = str(resume_["user_id"])
            resume_.pop("user_id")
            uid_to_resume[uid] = resume_

        for job in all_job_dict:
            job_ = job.copy()
            jd_no = str(job_["jd_no"])
            job_.pop("jd_no")
            jd_no_to_job[jd_no] = job_

        # prepare data
        data = []
        for label_data in label_pairs:
            resume_id = str(label_data["user_id"])
            job_id = str(label_data["jd_no"])
            label = int(label_data["satisfied"])

            resume = uid_to_resume[resume_id]
            job = jd_no_to_job[job_id]
            data.append((resume, job, label))
        return data

    def _encode_single_dict(self, dict_data: Dict[str, str], type: str):
        keys_to_encode = self.resume_key_names if type == 'resume' else self.job_key_names
        keys_to_encode_set = set(keys_to_encode)
        for k, v in dict_data.items():
            assert(k in keys_to_encode_set)

        encoded_dict_ = []
        for k in keys_to_encode:
            v = dict_data[k]
            v = str(v)

            field_max_len = self.max_seq_length_per_key[k]
            self.tokenizer_args["max_length"] = field_max_len
            # see https://huggingface.co/intfloat/multilingual-e5-large
            if self.query_prefix != "":
                k = f'{self.query_prefix}: {k}'
                v = f'{self.query_prefix}: {v}'
            encoded_value = self.tokenizer([v], **self.tokenizer_args)

            self.tokenizer_args["max_length"] = self.max_key_seq_length
            encoded_key_name = self.tokenizer([k], **self.tokenizer_args)

            encoded_dict_.append((
                k,
                {
                    "encoded_key_name": encoded_key_name,
                    "encoded_values": encoded_value,
                }
            ))
        # since Python 3.6, dicts will keep the insertion order
        # so as long as in the beginning they are the same, they will be the same
        encoded_dict = OrderedDict(encoded_dict_)
        return encoded_dict

    def _encode_single_taxon(self, taxon_token: str):
        taxon_token = taxon_token.strip()
        if taxon_token != "":
            if self.query_prefix != "":
                taxon_token = f'{self.query_prefix}: {taxon_token}'
            taxon_encoding = self.tokenizer(
                [taxon_token],
                # padding="do_not_pad",
                padding="max_length",  # to be consistent with InEXIT in the train_inexit_old.py
                return_tensors="pt",
                max_length=self.max_key_seq_length,
                truncation=True,
            )
        else:
            taxon_encoding = {}
        return taxon_encoding

    def encode_data(
        self,
        data: List[Tuple[dict, dict, int]],
        resume_taxon_token: str,
        job_taxon_token: str,
    ):
        """
        encode data to tensors
        """
        encoded_data = []
        for resume, job, label in tqdm(data, desc="Encoding data"):
            encoded_resume = self._encode_single_dict(resume, type='resume')
            encoded_job = self._encode_single_dict(job, type='job')

            encoded_resume_taxon = self._encode_single_taxon(resume_taxon_token)
            encoded_job_taxon = self._encode_single_taxon(job_taxon_token)

            encoded_data.append(
                {
                    "resume": encoded_resume,
                    "job": encoded_job,
                    "resume_taxon_encoding": encoded_resume_taxon,
                    "job_taxon_encoding": encoded_job_taxon,
                    "label": label,
                }
            )
        return encoded_data

    def __len__(self):
        return len(self.encoded_data)

    def __getitem__(self, index):
        return self.encoded_data[index]


class RJPairSimplifiedDataset(Dataset):
    """
    encode data to the format of
    a dict resume/job would become: {
        "desired_city": {
            "encoded_key_values": {},  # BatchEncoding of taxon_token + desired_city + value of "desired_city". Shape token_id=1 * seq_length
        },
        ...
    }
    """

    def __init__(
        self,
        tokenizer,
        max_seq_length_per_key: Dict[str, int],
        resume_key_names: List[str],
        job_key_names: List[str],
        tokenizer_args: Dict[str, Any],
        all_resume_dict: List[dict],
        all_job_dict: List[dict],
        label_pairs: List[Tuple[str, str, int]],
        resume_taxon_token: str = "",
        job_taxon_token: str = "",
        query_prefix: str = "",
    ):
        print("Using query_prefix:", query_prefix)

        self.tokenizer = tokenizer
        self.query_prefix = query_prefix  # used by special encoders such as e5
        self.max_seq_length_per_key = (
            max_seq_length_per_key  # will override the max_seq_length in tokenizer_args
        )
        self.resume_key_names = resume_key_names  # ensure the order of dict key is constant
        self.job_key_names = job_key_names
        self.resume_taxon_token = resume_taxon_token
        self.job_taxon_token = job_taxon_token
        self.tokenizer_args = tokenizer_args

        self.data = self.contruct_labeled_pairs(
            all_resume_dict, all_job_dict, label_pairs
        )
        self.encoded_data = self.encode_data(self.data)
        return

    def contruct_labeled_pairs(
        self,
        all_resume_dict: List[dict],
        all_job_dict: List[dict],
        label_pairs: List[dict],
    ):
        """
        label_pairs: [{'user_id': "xxx", 'jd_no': "xxx", 'satisifed': 0}, ...]
        output: [(resume_dict, job_dict, label), ...]
        """
        uid_to_resume = {}
        jd_no_to_job = {}

        for resume in all_resume_dict:
            resume_ = resume.copy()
            uid = str(resume_["user_id"])
            resume_.pop("user_id")
            uid_to_resume[uid] = resume_

        for job in all_job_dict:
            job_ = job.copy()
            jd_no = str(job_["jd_no"])
            job_.pop("jd_no")
            jd_no_to_job[jd_no] = job_

        # prepare data
        data = []
        for label_data in label_pairs:
            resume_id = str(label_data["user_id"])
            job_id = str(label_data["jd_no"])
            label = int(label_data["satisfied"])

            resume = uid_to_resume[resume_id]
            job = jd_no_to_job[job_id]
            data.append((resume, job, label))
        return data

    def _encode_single_dict(self, dict_data: Dict[str, str], type: str):
        keys_to_encode = self.resume_key_names if type == 'resume' else self.job_key_names
        keys_to_encode_set = set(keys_to_encode)
        for k, v in dict_data.items():
            assert(k in keys_to_encode_set)

        taxon_token = self.resume_taxon_token if type == 'resume' else self.job_taxon_token

        encoded_dict_ = []
        for k in keys_to_encode:
            v = dict_data[k]
            content = f"{taxon_token}: section {k}. {v}"

            field_max_len = self.max_seq_length_per_key[k]
            self.tokenizer_args["max_length"] = field_max_len
            # see https://huggingface.co/intfloat/multilingual-e5-large
            if self.query_prefix != "":
                content = f'{self.query_prefix}: {content}'
            encoded_kv = self.tokenizer([content], **self.tokenizer_args)

            encoded_dict_.append((
                k,
                {
                    "encoded_key_values": encoded_kv,
                }
            ))
        # since Python 3.6, dicts will keep the insertion order
        # so as long as in the beginning they are the same, they will be the same
        encoded_dict = OrderedDict(encoded_dict_)
        return encoded_dict

    def encode_data(
        self,
        data: List[Tuple[dict, dict, int]],
    ):
        """
        encode data to tensors
        """
        encoded_data = []
        for resume, job, label in tqdm(data, desc="Encoding data"):
            encoded_resume = self._encode_single_dict(resume, type='resume')
            encoded_job = self._encode_single_dict(job, type='job')

            encoded_data.append(
                {
                    "resume": encoded_resume,
                    "job": encoded_job,
                    "label": label,
                }
            )
        return encoded_data

    def __len__(self):
        return len(self.encoded_data)

    def __getitem__(self, index):
        return self.encoded_data[index]


class RJPair2DSimplifiedDataset(Dataset):
    """
    encode data to the format of
    a dict resume/job would become: {
        "resume_sents": { # BatchEncoding of k sentences from a resume
            "input_ids": torch.tensor,
        },
        "job_sents": { # BatchEncoding of k sentences from a job
            "input_ids": torch.tensor,
        },
        "label": torch.tensor,  # shape batch_size
        ...
    }
    """

    def __init__(
        self,
        tokenizer,
        max_seq_len: int,
        resume_key_names: List[str],
        job_key_names: List[str],
        tokenizer_args: Dict[str, Any],
        all_resume_dict: List[dict],
        all_job_dict: List[dict],
        label_pairs: List[Tuple[str, str, int]],
        resume_taxon_token: str = "",
        job_taxon_token: str = "",
        query_prefix: str = "",
    ):
        print("Using query_prefix:", query_prefix)

        self.tokenizer = tokenizer
        self.query_prefix = query_prefix  # used by special encoders such as e5
        self.max_seq_len = max_seq_len
        self.resume_key_names = resume_key_names  # ensure the order of dict key is constant
        self.job_key_names = job_key_names
        self.resume_taxon_token = resume_taxon_token
        self.job_taxon_token = job_taxon_token
        self.tokenizer_args = tokenizer_args
        self.tokenizer_args["max_length"] = self.max_seq_len

        self.data = self.contruct_labeled_pairs(
            all_resume_dict, all_job_dict, label_pairs
        )
        self.encoded_data = self.encode_data(self.data)
        return

    def contruct_labeled_pairs(
        self,
        all_resume_dict: List[dict],
        all_job_dict: List[dict],
        label_pairs: List[dict],
    ):
        """
        label_pairs: [{'user_id': "xxx", 'jd_no': "xxx", 'satisifed': 0}, ...]
        output: [(resume_dict, job_dict, label), ...]
        """
        uid_to_resume = {}
        jd_no_to_job = {}

        for resume in all_resume_dict:
            resume_ = resume.copy()
            uid = str(resume_["user_id"])
            resume_.pop("user_id")
            uid_to_resume[uid] = resume_

        for job in all_job_dict:
            job_ = job.copy()
            jd_no = str(job_["jd_no"])
            job_.pop("jd_no")
            jd_no_to_job[jd_no] = job_

        # prepare data
        data = []
        for label_data in label_pairs:
            resume_id = str(label_data["user_id"])
            job_id = str(label_data["jd_no"])
            label = int(label_data["satisfied"])

            resume = uid_to_resume[resume_id]
            job = jd_no_to_job[job_id]
            data.append((resume, job, label))
        return data

    def _encode_single_dict(self, dict_data: Dict[str, str], type: str):
        keys_to_encode = self.resume_key_names if type == 'resume' else self.job_key_names
        keys_to_encode_set = set(keys_to_encode)
        for k, v in dict_data.items():
            assert(k in keys_to_encode_set)

        taxon_token = self.resume_taxon_token if type == 'resume' else self.job_taxon_token

        lines_to_encode = []
        for k in keys_to_encode:
            v = dict_data[k]
            content = f"{taxon_token}: section {k}. {v}"
            lines_to_encode.append(content)
        
        encoded_lines = self.tokenizer(lines_to_encode, **self.tokenizer_args)
        return encoded_lines

    def encode_data(
        self,
        data: List[Tuple[dict, dict, int]],
    ):
        """
        encode data to tensors
        """
        encoded_data = []
        for resume, job, label in tqdm(data, desc="Encoding data"):
            encoded_resume = self._encode_single_dict(resume, type='resume')
            encoded_job = self._encode_single_dict(job, type='job')

            encoded_data.append(
                {
                    "resume_sents": encoded_resume,
                    "job_sents": encoded_job,
                    "label": label,
                }
            )
        return encoded_data

    def __len__(self):
        return len(self.encoded_data)

    def __getitem__(self, index):
        return self.encoded_data[index]


class RJPairNewContrastiveDataset(Dataset):
    """
    encode data to the format of
    a dict resume/job would become: {
        "desired_city": {
            "encoded_key_values": {},  # BatchEncoding of taxon_token + desired_city + value of "desired_city". Shape token_id=1 * seq_length
        },
        ...
    }
    but additionally return, for each resume-job pair:
    - list of at most 3? hard negative resume for that job
    - list of at most 3? hard negative job for that resume
    """

    def __init__(
        self,
        tokenizer,
        max_seq_length_per_key: Dict[str, int],
        resume_key_names: List[str],
        job_key_names: List[str],
        tokenizer_args: Dict[str, Any],
        all_resume_dict: List[dict],
        all_job_dict: List[dict],
        label_pairs: List[dict],
        resume_taxon_token: str = "",
        job_taxon_token: str = "",
        query_prefix: str = "",
    ):
        print("Using query_prefix:", query_prefix)

        self.tokenizer = tokenizer
        self.query_prefix = query_prefix  # used by special encoders such as e5
        self.max_seq_length_per_key = (
            max_seq_length_per_key  # will override the max_seq_length in tokenizer_args
        )
        self.resume_key_names = resume_key_names  # ensure the order of dict key is constant
        self.job_key_names = job_key_names
        self.resume_taxon_token = resume_taxon_token
        self.job_taxon_token = job_taxon_token
        self.tokenizer_args = tokenizer_args

        self.resume_df = pd.DataFrame(all_resume_dict)
        self.resume_df.index = self.resume_df["user_id"].astype(str).values
        self.job_df = pd.DataFrame(all_job_dict)
        self.job_df.index = self.job_df["jd_no"].astype(str).values

        self.r_to_j_mappping, self.j_to_r_mapping = self.construct_rj_mappings(label_pairs)
        self.positive_pairs = []
        for label_data in label_pairs:
            label = int(label_data["satisfied"])
            if label == 1:
                self.positive_pairs.append(label_data)
        return

    def construct_rj_mappings(self, label_pairs: List[Tuple[str, str, int]],):
        r_to_j_mappping = {}
        j_to_r_mapping = {}
        for label_data in label_pairs:
            resume_id = str(label_data["user_id"])
            job_id = str(label_data["jd_no"])
            label = int(label_data["satisfied"])

            if resume_id not in r_to_j_mappping:
                r_to_j_mappping[resume_id] = {
                    'pos_j': set(),
                    'neg_j': set(),
                }
            if job_id not in j_to_r_mapping:
                j_to_r_mapping[job_id] = {
                    'pos_r': set(),
                    'neg_r': set(),
                }

            if label == 1:
                r_to_j_mappping[resume_id]['pos_j'].add(job_id)
                j_to_r_mapping[job_id]['pos_r'].add(resume_id)
            else:
                r_to_j_mappping[resume_id]['neg_j'].add(job_id)
                j_to_r_mapping[job_id]['neg_r'].add(resume_id)
        return r_to_j_mappping, j_to_r_mapping

    def _encode_single_dict(self, dict_data: Dict[str, str], type: str):
        keys_to_encode = self.resume_key_names if type == 'resume' else self.job_key_names
        keys_to_encode_set = set(keys_to_encode)
        for k, v in dict_data.items():
            assert(k in keys_to_encode_set)

        taxon_token = self.resume_taxon_token if type == 'resume' else self.job_taxon_token

        encoded_dict_ = []
        for k in keys_to_encode:
            v = dict_data[k]
            content = f"{taxon_token}: section {k}. {v}"

            field_max_len = self.max_seq_length_per_key[k]
            self.tokenizer_args["max_length"] = field_max_len
            # see https://huggingface.co/intfloat/multilingual-e5-large
            if self.query_prefix != "":
                content = f'{self.query_prefix}: {content}'
            encoded_kv = self.tokenizer([content], **self.tokenizer_args)

            encoded_dict_.append((
                k,
                {
                    "encoded_key_values": encoded_kv,
                }
            ))
        # since Python 3.6, dicts will keep the insertion order
        # so as long as in the beginning they are the same, they will be the same
        encoded_dict = OrderedDict(encoded_dict_)
        return encoded_dict

    def __len__(self):
        return len(self.positive_pairs)

    def __getitem__(self, index):
        pair = self.positive_pairs[index]
        resume_id = str(pair["user_id"])
        job_id = str(pair["jd_no"])

        resume = self.resume_df.loc[resume_id].to_dict()
        resume.pop("user_id")
        encoded_resume = self._encode_single_dict(resume, type='resume')
        job = self.job_df.loc[job_id].to_dict()
        job.pop("jd_no")
        encoded_job = self._encode_single_dict(job, type='job')

        # get hard negatives
        hard_negative_resume_ids = self.j_to_r_mapping[job_id]['neg_r']
        hard_negatives_resumes_encoded = []
        for user_id in hard_negative_resume_ids:
            resume = self.resume_df.loc[user_id].to_dict()
            resume.pop("user_id")
            hard_negatives_resumes_encoded.append(
                self._encode_single_dict(resume, type='resume')
            )
        hard_negative_job_ids = self.r_to_j_mappping[resume_id]['neg_j']
        hard_negatives_jobs_encoded = []
        for jd_no in hard_negative_job_ids:
            job = self.job_df.loc[jd_no].to_dict()
            job.pop("jd_no")
            hard_negatives_jobs_encoded.append(
                self._encode_single_dict(job, type='job')
            )

        # get positives. A resume is similar to resume that got the same job, and a job is similar to job that accepted the same resume
        positive_resume_ids = self.j_to_r_mapping[job_id]['pos_r']
        positive_resume_encoded = []
        for pos_resume_id in positive_resume_ids:
            if pos_resume_id == resume_id:
                continue
            resume = self.resume_df.loc[pos_resume_id].to_dict()
            resume.pop("user_id")
            positive_resume_encoded.append(
                self._encode_single_dict(resume, type='resume')
            )
        positive_job_ids = self.r_to_j_mappping[resume_id]['pos_j']
        positive_job_encoded = []
        for pos_job_id in positive_job_ids:
            if pos_job_id == job_id:
                continue
            job = self.job_df.loc[pos_job_id].to_dict()
            job.pop("jd_no")
            positive_job_encoded.append(
                self._encode_single_dict(job, type='job')
            )
        return {
            'resume': encoded_resume,
            'job': encoded_job,
            'resume_hard_negatives': hard_negatives_resumes_encoded,
            'job_hard_negatives': hard_negatives_jobs_encoded,
            'resume_positives': positive_resume_encoded,
            'job_positives': positive_job_encoded,
        }


class RJPairNewContrastivewAugDataset(Dataset):
    """
    encode data to the format of
    a dict resume/job would become: {
        "desired_city": {
            "encoded_key_values": {},  # BatchEncoding of taxon_token + desired_city + value of "desired_city". Shape token_id=1 * seq_length
        },
        ...
    }
    but additionally return, for each resume-job pair:
    - list of at most 3? hard negative resume for that job
    - list of at most 3? hard negative job for that resume
    - add data augmentation during sampling (this means resume is no longer flattened)
    Implementation-wise: a resume/job will ALWAYS pass through the augmentation function. Inside that function, you can decide whether to augment or not.
    """

    def __init__(
        self,
        tokenizer,
        max_seq_length_per_key: Dict[str, int],
        resume_key_names: List[str],
        job_key_names: List[str],
        tokenizer_args: Dict[str, Any],
        all_resume_dict: List[dict],
        all_job_dict: List[dict],
        label_pairs: List[dict],
        resume_aug_fn: Callable[[dict], dict],
        job_aug_fn: Callable[[dict], dict],
        no_aug: bool = False,
        resume_taxon_token: str = "",
        job_taxon_token: str = "",
        query_prefix: str = "",
    ):
        print("Using query_prefix:", query_prefix)

        self.tokenizer = tokenizer
        self.query_prefix = query_prefix  # used by special encoders such as e5
        self.max_seq_length_per_key = (
            max_seq_length_per_key  # will override the max_seq_length in tokenizer_args
        )
        self.resume_key_names = resume_key_names  # ensure the order of dict key is constant
        self.job_key_names = job_key_names
        self.resume_taxon_token = resume_taxon_token
        self.job_taxon_token = job_taxon_token
        self.tokenizer_args = tokenizer_args

        self.resume_df = pd.DataFrame(all_resume_dict)
        self.resume_df.index = self.resume_df["user_id"].astype(str).values
        self.job_df = pd.DataFrame(all_job_dict)
        self.job_df.index = self.job_df["jd_no"].astype(str).values

        self.resume_aug_fn = resume_aug_fn
        self.job_aug_fn = job_aug_fn
        # if True, then no augmentation is done (controlled by the augmentation function)
        # why are we passing this to the function? Because for proprietary data we still need it to flatten it
        self.no_aug = no_aug

        self.r_to_j_mappping, self.j_to_r_mapping = self.construct_rj_mappings(label_pairs)
        self.positive_pairs = []
        for label_data in label_pairs:
            label = int(label_data["satisfied"])
            if label == 1:
                self.positive_pairs.append(label_data)
        return

    def construct_rj_mappings(self, label_pairs: List[Tuple[str, str, int]],):
        r_to_j_mappping = {}
        j_to_r_mapping = {}
        for label_data in label_pairs:
            resume_id = str(label_data["user_id"])
            job_id = str(label_data["jd_no"])
            label = int(label_data["satisfied"])

            if resume_id not in r_to_j_mappping:
                r_to_j_mappping[resume_id] = {
                    'pos_j': set(),
                    'neg_j': set(),
                }
            if job_id not in j_to_r_mapping:
                j_to_r_mapping[job_id] = {
                    'pos_r': set(),
                    'neg_r': set(),
                }

            if label == 1:
                r_to_j_mappping[resume_id]['pos_j'].add(job_id)
                j_to_r_mapping[job_id]['pos_r'].add(resume_id)
            else:
                r_to_j_mappping[resume_id]['neg_j'].add(job_id)
                j_to_r_mapping[job_id]['neg_r'].add(resume_id)
        return r_to_j_mappping, j_to_r_mapping

    def _encode_single_dict(self, dict_data: Dict[str, str], type: str):
        keys_to_encode = self.resume_key_names if type == 'resume' else self.job_key_names
        keys_to_encode_set = set(keys_to_encode)
        for k, v in dict_data.items():
            assert(k in keys_to_encode_set)

        taxon_token = self.resume_taxon_token if type == 'resume' else self.job_taxon_token

        encoded_dict_ = []
        for k in keys_to_encode:
            v = dict_data[k]
            content = f"{taxon_token}: section {k}. {v}"

            field_max_len = self.max_seq_length_per_key[k]
            self.tokenizer_args["max_length"] = field_max_len
            # see https://huggingface.co/intfloat/multilingual-e5-large
            if self.query_prefix != "":
                content = f'{self.query_prefix}: {content}'
            encoded_kv = self.tokenizer([content], **self.tokenizer_args)

            encoded_dict_.append((
                k,
                {
                    "encoded_key_values": encoded_kv,
                }
            ))
        # since Python 3.6, dicts will keep the insertion order
        # so as long as in the beginning they are the same, they will be the same
        encoded_dict = OrderedDict(encoded_dict_)
        return encoded_dict

    def __len__(self):
        return len(self.positive_pairs)

    def __getitem__(self, index):
        pair = self.positive_pairs[index]
        resume_id = str(pair["user_id"])
        job_id = str(pair["jd_no"])

        resume = self.resume_df.loc[resume_id].to_dict()
        resume = self.resume_aug_fn(resume, data_type='noop' if self.no_aug else 'positive')
        resume.pop("user_id")
        encoded_resume = self._encode_single_dict(resume, type='resume')
        job = self.job_df.loc[job_id].to_dict()
        job = self.job_aug_fn(job, data_type='noop' if self.no_aug else 'positive')
        job.pop("jd_no")
        encoded_job = self._encode_single_dict(job, type='job')

        # get hard negatives
        hard_negative_resume_ids = self.j_to_r_mapping[job_id]['neg_r']
        hard_negatives_resumes_encoded = []
        for user_id in hard_negative_resume_ids:
            resume = self.resume_df.loc[user_id].to_dict()
            resume = self.resume_aug_fn(resume, data_type='noop' if self.no_aug else 'negative')
            resume.pop("user_id")
            hard_negatives_resumes_encoded.append(
                self._encode_single_dict(resume, type='resume')
            )
        hard_negative_job_ids = self.r_to_j_mappping[resume_id]['neg_j']
        hard_negatives_jobs_encoded = []
        for jd_no in hard_negative_job_ids:
            job = self.job_df.loc[jd_no].to_dict()
            job = self.job_aug_fn(job, data_type='noop' if self.no_aug else 'negative')
            job.pop("jd_no")
            hard_negatives_jobs_encoded.append(
                self._encode_single_dict(job, type='job')
            )

        # get positives. A resume is similar to resume that got the same job, and a job is similar to job that accepted the same resume
        positive_resume_ids = self.j_to_r_mapping[job_id]['pos_r']
        positive_resume_encoded = []
        for pos_resume_id in positive_resume_ids:
            if pos_resume_id == resume_id:
                continue
            resume = self.resume_df.loc[pos_resume_id].to_dict()
            resume = self.resume_aug_fn(resume, data_type='noop' if self.no_aug else 'positive')
            resume.pop("user_id")
            positive_resume_encoded.append(
                self._encode_single_dict(resume, type='resume')
            )
        positive_job_ids = self.r_to_j_mappping[resume_id]['pos_j']
        positive_job_encoded = []
        for pos_job_id in positive_job_ids:
            if pos_job_id == job_id:
                continue
            job = self.job_df.loc[pos_job_id].to_dict()
            job = self.job_aug_fn(job, data_type='noop' if self.no_aug else 'positive')
            job.pop("jd_no")
            positive_job_encoded.append(
                self._encode_single_dict(job, type='job')
            )
        return {
            'resume': encoded_resume,
            'job': encoded_job,
            'resume_hard_negatives': hard_negatives_resumes_encoded,
            'job_hard_negatives': hard_negatives_jobs_encoded,
            'resume_positives': positive_resume_encoded,
            'job_positives': positive_job_encoded,
        }


class RJPairPretrainwAugDataset(Dataset):
    """
    used for contrastive pretraining
    encode data to the format of
    a dict resume/job would become: {
        "desired_city": {
            "encoded_key_values": {},  # BatchEncoding of taxon_token + desired_city + value of "desired_city". Shape token_id=1 * seq_length
        },
        ...
    }
    but additionally return, for each resume-job pair:
    - list of at most 3? hard negative resume for that job
    - list of at most 3? hard negative job for that resume
    - add data augmentation during sampling (this means resume is no longer flattened)
    Implementation-wise: a resume/job will ALWAYS pass through the augmentation function. Inside that function, you can decide whether to augment or not.
    """

    def __init__(
        self,
        tokenizer,
        max_seq_length_per_key: Dict[str, int],
        resume_key_names: List[str],
        job_key_names: List[str],
        tokenizer_args: Dict[str, Any],
        all_resume_dict: List[dict],
        all_job_dict: List[dict],
        resume_aug_fn: Callable[[dict], dict],
        job_aug_fn: Callable[[dict], dict],
        resume_taxon_token: str = "",
        job_taxon_token: str = "",
        query_prefix: str = "",
        randomize: bool = True,
    ):
        print("Using query_prefix:", query_prefix)

        self.tokenizer = tokenizer
        self.query_prefix = query_prefix  # used by special encoders such as e5
        self.max_seq_length_per_key = (
            max_seq_length_per_key  # will override the max_seq_length in tokenizer_args
        )
        self.resume_key_names = resume_key_names  # ensure the order of dict key is constant
        self.job_key_names = job_key_names
        self.resume_taxon_token = resume_taxon_token
        self.job_taxon_token = job_taxon_token
        self.tokenizer_args = tokenizer_args

        self.resume_aug_fn = resume_aug_fn
        self.job_aug_fn = job_aug_fn

        self.num_r = len(all_resume_dict)
        self.all_data = all_resume_dict + all_job_dict
        
        self.randomize = randomize
        return

    def _encode_single_dict(self, dict_data: Dict[str, str], type: str):
        keys_to_encode = self.resume_key_names if type == 'resume' else self.job_key_names
        keys_to_encode_set = set(keys_to_encode)
        for k, v in dict_data.items():
            assert(k in keys_to_encode_set)

        taxon_token = self.resume_taxon_token if type == 'resume' else self.job_taxon_token

        encoded_dict_ = []
        for k in keys_to_encode:
            v = dict_data[k]
            content = f"{taxon_token}: section {k}. {v}"

            field_max_len = self.max_seq_length_per_key[k]
            self.tokenizer_args["max_length"] = field_max_len
            # see https://huggingface.co/intfloat/multilingual-e5-large
            if self.query_prefix != "":
                content = f'{self.query_prefix}: {content}'
            encoded_kv: BatchEncoding = self.tokenizer([content], **self.tokenizer_args)
            # squeeze
            encoded_kv_squeezed = {
                "input_ids": encoded_kv.input_ids.squeeze(0),
                "attention_mask": encoded_kv.attention_mask.squeeze(0),
            }
            if hasattr(encoded_kv, "token_type_ids"):
                encoded_kv_squeezed["token_type_ids"] = encoded_kv.token_type_ids.squeeze(0)
            encoded_kv_squeezed = BatchEncoding(data=encoded_kv_squeezed)

            encoded_dict_.append((
                k, encoded_kv_squeezed
            ))
        # since Python 3.6, dicts will keep the insertion order
        # so as long as in the beginning they are the same, they will be the same
        encoded_dict = OrderedDict(encoded_dict_)
        return encoded_dict

    def __len__(self):
        return len(self.all_data)

    def __getitem__(self, index):
        # return a tuple of (view1, view2, data_type)
        if self.randomize:
            # randomize the index but keep the same data type
            if index < self.num_r:
                index = random.randint(0, self.num_r - 1)
            else:
                index = random.randint(self.num_r, len(self.all_data) - 1)
        
        sample_data = self.all_data[index]
        if 'user_id' in sample_data:
            data_type = 'resume'
            view_1 = self.resume_aug_fn(sample_data)
            view_2 = self.resume_aug_fn(sample_data)
            view_1.pop("user_id")
            view_2.pop("user_id")
            return (
                self._encode_single_dict(view_1, type='resume'),
                self._encode_single_dict(view_2, type='resume'),
                data_type,
            )
        else:
            data_type = 'job'
            view_1 = self.job_aug_fn(sample_data)
            view_2 = self.job_aug_fn(sample_data)
            view_1.pop("jd_no")
            view_2.pop("jd_no")
            return (
                self._encode_single_dict(view_1, type='job'),
                self._encode_single_dict(view_2, type='job'),
                data_type,
            )


def _recursive_batch_tensors(data: dict, batched_data: dict):
    for k, v in data.items():
        # initialization
        if k not in batched_data:
            if k in ["input_ids", "token_type_ids", "attention_mask", "label"]:
                batched_data[k] = []
            else:
                batched_data[k] = {}
        # base case
        if isinstance(v, (torch.Tensor, int, float)):
            batched_data[k].append(v)
        # recursive case
        elif isinstance(v, dict):
            _recursive_batch_tensors(v, batched_data[k])
        elif isinstance(v, BatchEncoding):
            _recursive_batch_tensors(v, batched_data[k])
            # and format it back to BatchEncoding
            batched_data[k] = BatchEncoding(batched_data[k])
        else:
            raise ValueError(f"Unknown type {type(v)}")
    return


def _recursive_to_tensors(data: dict):
    # convert all the list of tensors or int/floats to tensors
    for k, v in data.items():
        # base case
        if isinstance(v, list):
            if isinstance(v[0], torch.Tensor):
                data[k] = torch.concatenate(v, dim=0)
            else:
                data[k] = torch.tensor(v)
        elif isinstance(v, (dict, BatchEncoding)):  # recursive case
            _recursive_to_tensors(v)
        else:
            raise ValueError(f"Unknown type {type(v)}")
    return


def rj_pair_collate_fn(batch: list):
    """
    merge a list of dict, such that it has the same key: value structure, but
    whenever we see token_ids etc we will batch them to be like:
    batch it to be like:
    {
        'batched_resume': {
            key_name: {
                "encoded_key_name": {},  # shape batch_size * seq_length
                "encoded_values": {},  # shape batch_size * seq_length
            },
        }
    }
    """
    batched_dict = {}
    for b in batch:
        _recursive_batch_tensors(b, batched_dict)
    _recursive_to_tensors(batched_dict)

    batched_dict_updated_name = {}
    for k, v in batched_dict.items():
        new_key_name = f"batched_{k}"
        batched_dict_updated_name[new_key_name] = v
    return batched_dict_updated_name


def contrastive_rj_pair_collate_fn(batch: list, max_num_hard_negatives: int):
    """
    resume_negative_prob: probability that this batch is doing resume_hard_negatives

    merge a list of dict, such that we return like
    {
        'batched_resume': {
            resume_key_name: {
                "encoded_key_name": {},  # shape batch_size * seq_length
                "encoded_values": {},  # shape batch_size * seq_length
            },
        },
        'batched_job': {
            job_key_name: {
                "encoded_key_name": {},  # shape batch_size * seq_length
                "encoded_values": {},  # shape batch_size * seq_length
            },
        }
        # if it is batched_resume_hard_negatives
        'batched_resume_hard_negatives': {
            resume_key_name: {
                "encoded_key_name": {},  # shape batch_size * seq_length
                "encoded_values": {},  # shape batch_size * seq_length
            },
        }
    }
    """
    batched_dict = {}

    # if one of them is empty, then hard_negative non-empty one
    all_can_sample_resume = []
    all_can_sample_job = []
    for b in batch:
        for neg in b["resume_hard_negatives"]:
            # do it like this so we can recycle code to use rj_pair_collate_fn
            all_can_sample_resume.append({
                "resume_hard_negatives": neg,
            })
        for neg in b["job_hard_negatives"]:
            all_can_sample_job.append({
                "job_hard_negatives": neg,
            })

    sample_neg_resume = True
    if len(all_can_sample_resume) == 0:
        sample_neg_resume = False
    elif len(all_can_sample_resume) > 0 and len(all_can_sample_job) > 0:
        sample_neg_resume = random.random() < 0.5

    # first, construct the negative samples manually
    if sample_neg_resume:
        # do resume hard negatives for JOBs
        # for now we just PRAY we will not sample positive resumes
        # TODO: do it sample ids from resume_positive_ids
        num_to_sample = min(max_num_hard_negatives, len(all_can_sample_resume))
        if num_to_sample > 0:
            sampled_resume_hard_negs = random.sample(all_can_sample_resume, num_to_sample)
            batched_resume_hard_negatives = rj_pair_collate_fn(sampled_resume_hard_negs)
            batched_dict['batched_resume_hard_negatives'] = batched_resume_hard_negatives['batched_resume_hard_negatives']
    else:
        # do job hard negatives for RESUMEs
        num_to_sample = min(max_num_hard_negatives, len(all_can_sample_job))
        if num_to_sample > 0:
            sampled_job_hard_negs = random.sample(all_can_sample_job, num_to_sample)
            batched_job_hard_negatives = rj_pair_collate_fn(sampled_job_hard_negs)
            batched_dict['batched_job_hard_negatives'] = batched_job_hard_negatives['batched_job_hard_negatives']
    
    # cleaning
    positive_pairs = []
    for b in batch:
        positive_pairs.append({
            'resume': b['resume'],
            'job': b['job'],
        })
    positive_pairs_batched = rj_pair_collate_fn(positive_pairs)
    batched_dict['batched_resume'] = positive_pairs_batched['batched_resume']
    batched_dict['batched_job'] = positive_pairs_batched['batched_job']
    return batched_dict


def rj_pair_w_symm_negatives_collate_fn(batch: list):
    """
    merge a list of dict, such that we return like
    {
        'batched_resume': {
            resume_key_name: {
                "encoded_key_name": {},  # shape batch_size * seq_length
                "encoded_values": {},  # shape batch_size * seq_length
            },
        },
        'batched_job': {
            job_key_name: {
                "encoded_key_name": {},  # shape batch_size * seq_length
                "encoded_values": {},  # shape batch_size * seq_length
            },
        }
        # match the size of batched_resume
        'batched_resume_hard_negatives': {
            resume_key_name: {
                "encoded_key_name": {},  # shape batch_size * seq_length
                "encoded_values": {},  # shape batch_size * seq_length
            },
        },
        # match the size of batched_job
        'batched_job_hard_negatives': {
            job_key_name: {
                "encoded_key_name": {},  # shape batch_size * seq_length
                "encoded_values": {},  # shape batch_size * seq_length
            },
        }
    }
    """
    batched_dict = {}

    # if one of them is empty, pad with random ones in this batch
    neg_resume = []
    neg_job = []
    for idx, b in enumerate(batch):
        other_than_this_batch = batch[:idx] + batch[idx+1:]
        if len(b["resume_hard_negatives"]) == 0:
            # randomly sample one from this batch
            neg_resume.append({
                "resume_negatives": random.choice(other_than_this_batch)["resume"],
            })
        else:
            neg_resume.append({
                "resume_negatives": random.choice(b["resume_hard_negatives"]),
            })
        if len(b["job_hard_negatives"]) == 0:
            # randomly sample one from this batch
            neg_job.append({
                "job_negatives": random.choice(other_than_this_batch)["job"],
            })
        else:
            neg_job.append({
                "job_negatives": random.choice(b["job_hard_negatives"]),
            })
    batched_resume_negatives = rj_pair_collate_fn(neg_resume)
    batched_dict['batched_resume_negatives'] = batched_resume_negatives['batched_resume_negatives']
    batched_job_negatives = rj_pair_collate_fn(neg_job)
    batched_dict['batched_job_negatives'] = batched_job_negatives['batched_job_negatives']

    # cleaning
    positive_pairs = []
    for b in batch:
        positive_pairs.append({
            'resume': b['resume'],
            'job': b['job'],
        })
    positive_pairs_batched = rj_pair_collate_fn(positive_pairs)
    batched_dict['batched_resume'] = positive_pairs_batched['batched_resume']
    batched_dict['batched_job'] = positive_pairs_batched['batched_job']
    return batched_dict


def contrastive_rj_pair_w_indicies_collate_fn(batch: list, max_num_hard_negatives: int):
    """
    merge a list of dict, such that we return like
    {
        'batched_resume': {
            resume_key_name: {
                "encoded_key_name": {},  # shape batch_size * seq_length
                "encoded_values": {},  # shape batch_size * seq_length
            },
        },
        'batched_job': {
            job_key_name: {
                "encoded_key_name": {},  # shape batch_size * seq_length
                "encoded_values": {},  # shape batch_size * seq_length
            },
        }
        # if it is batched_resume_hard_negatives
        'batched_resume_hard_negatives': {
            resume_key_name: {
                "encoded_key_name": {},  # shape batch_size * seq_length
                "encoded_values": {},  # shape batch_size * seq_length
            },
        },
        # if it is batched_resume_hard_negatives, the paired job index
        'batched_hard_negatives_indices': [],
    }
    """
    batched_dict = {}
    # TODO: p(sample_resume | has_neg_resume) = (1 - p(sample_resume | has_neg_resume)) * job_to_resume_ratio
    
    # for hard_negatives
    all_can_sample_resume = []
    all_can_sample_resume_paired_job_indices = []
    all_can_sample_job = []
    all_can_sample_job_paired_resume_indices = []
    # for positives
    all_can_sample_resume_positives = []
    all_can_sample_job_positives = []
    for idx, b in enumerate(batch):
        for neg in b["resume_hard_negatives"]:
            # do it like this so we can recycle code to use rj_pair_collate_fn
            all_can_sample_resume.append({
                "resume_hard_negatives": neg,
            })
            all_can_sample_resume_paired_job_indices.append(idx)
        for neg in b["job_hard_negatives"]:
            all_can_sample_job.append({
                "job_hard_negatives": neg,
            })
            all_can_sample_job_paired_resume_indices.append(idx)
        
        all_can_sample_resume_positives.append(b["resume_positives"])
        all_can_sample_job_positives.append(b["job_positives"])
    
    neg_resume_only = True  # back compatibility
    if len(all_can_sample_resume) == 0:
        neg_resume_only = False
    elif len(all_can_sample_resume) > 0 and len(all_can_sample_job) > 0:
        neg_resume_only = random.random() < 0.5

    # first, construct the negative samples manually
    # do resume hard negatives for JOBs
    # for now we just PRAY we will not sample positive resumes
    num_to_sample = min(max_num_hard_negatives, len(all_can_sample_resume))
    if num_to_sample > 0:
        sampled_resume_hard_negs_idx = random.sample(range(len(all_can_sample_resume)), num_to_sample)
        sampled_resume_hard_negs = [all_can_sample_resume[i] for i in sampled_resume_hard_negs_idx]
        sampled_paired_job_indices = [all_can_sample_resume_paired_job_indices[i] for i in sampled_resume_hard_negs_idx]
        batched_resume_hard_negatives = rj_pair_collate_fn(sampled_resume_hard_negs)
        batched_dict['batched_resume_hard_negatives'] = batched_resume_hard_negatives['batched_resume_hard_negatives']
        batched_dict['batched_resume_hard_negatives_indices'] = sampled_paired_job_indices
    
    # do job hard negatives for RESUMEs
    num_to_sample = min(max_num_hard_negatives, len(all_can_sample_job))
    if num_to_sample > 0:
        sampled_job_hard_negs_idx = random.sample(range(len(all_can_sample_job)), num_to_sample)
        sampled_job_hard_negs = [all_can_sample_job[i] for i in sampled_job_hard_negs_idx]
        sampled_paired_resume_indices = [all_can_sample_job_paired_resume_indices[i] for i in sampled_job_hard_negs_idx]
        batched_job_hard_negatives = rj_pair_collate_fn(sampled_job_hard_negs)
        batched_dict['batched_job_hard_negatives'] = batched_job_hard_negatives['batched_job_hard_negatives']
        batched_dict['batched_job_hard_negatives_indices'] = sampled_paired_resume_indices

    # do the positive resumes for resume. We sample ONE per index
    resume_positive_indices = []
    sampled_resume_positives = []
    for i, pos_resumes in enumerate(all_can_sample_resume_positives):
        if len(pos_resumes) == 0:
            continue
        resume_positive_indices.append(i)
        random_pos_resume = random.choice(pos_resumes)
        sampled_resume_positives.append({
            "resume_positives": random_pos_resume,
        })
    if len(sampled_resume_positives) > 0:
        batched_resume_positives = rj_pair_collate_fn(sampled_resume_positives)
        batched_dict['batched_resume_positives'] = batched_resume_positives['batched_resume_positives']
        batched_dict['batched_resume_positive_indices'] = resume_positive_indices

    # do the positive jobs for job. We sample ONE per index
    job_positive_indices = []
    sampled_job_positives = []
    for i, pos_jobs in enumerate(all_can_sample_job_positives):
        if len(pos_jobs) == 0:
            continue
        job_positive_indices.append(i)
        random_pos_job = random.choice(pos_jobs)
        sampled_job_positives.append({
            "job_positives": random_pos_job,
        })
    if len(sampled_job_positives) > 0:
        batched_job_positives = rj_pair_collate_fn(sampled_job_positives)
        batched_dict['batched_job_positives'] = batched_job_positives['batched_job_positives']
        batched_dict['batched_job_positive_indices'] = job_positive_indices
    
    # cleaning
    positive_pairs = []
    for b in batch:
        positive_pairs.append({
            'resume': b['resume'],
            'job': b['job'],
        })
    positive_pairs_batched = rj_pair_collate_fn(positive_pairs)
    batched_dict['batched_resume'] = positive_pairs_batched['batched_resume']
    batched_dict['batched_job'] = positive_pairs_batched['batched_job']
    batched_dict['neg_resume_only'] = neg_resume_only
    return batched_dict


def __pretrain_collate_view_data(batch: list):
    collated_dict_data = {}
    num_batches = len(batch)
    if num_batches == 0:
        return collated_dict_data
    
    for k in batch[0].keys():
        _tmp: List[BatchEncoding] = []
        for i in range(num_batches):
            _tmp.append(batch[i][k])
        collated_dict_data[k] = default_data_collator(_tmp)
    return collated_dict_data


def pretrain_collate_view_data(batch: list):
    resume_view_1 = []
    resume_view_2 = []
    job_view_1 = []
    job_view_2 = []
    for data in batch:
        view_1, view_2, data_type = data
        if data_type == "resume":
            resume_view_1.append(view_1)
            resume_view_2.append(view_2)
        else:
            job_view_1.append(view_1)
            job_view_2.append(view_2)
    # collate
    resume_view_1_collated = __pretrain_collate_view_data(resume_view_1)
    resume_view_2_collated = __pretrain_collate_view_data(resume_view_2)
    job_view_1_collated = __pretrain_collate_view_data(job_view_1)
    job_view_2_collated = __pretrain_collate_view_data(job_view_2)
    return resume_view_1_collated, resume_view_2_collated, job_view_1_collated, job_view_2_collated