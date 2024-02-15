# configruation
from copy import deepcopy
from typing import Callable


_resume_col_name_mapping = {
    "live_city_id": "居住城市",
    "desire_jd_city_id": "期望工作城市",
    "desire_jd_industry_id": "期望工作行业",
    "desire_jd_type_id": "期望工作类型",
    "desire_jd_salary_id": "期望薪资",
    "cur_industry_id": "当前工作行业",
    "cur_jd_type": "当前工作类型",
    "cur_salary_id": "当前薪资",
    "cur_degree_id": "学历",
    "birthday": "年龄",
    "start_work_date": "开始工作时间",
    "experience": "工作经验",
}

_jd_col_name_mapping = {
    "jd_title": "工作名称",
    "city": "工作城市",
    "jd_sub_type": "工作类型",
    "require_nums": "招聘人数",
    "salary": "薪资",
    "start_date": "招聘开始时间",
    "end_date": "招聘结束时间",
    "is_travel": "是否要求出差",
    "min_years": "工作年限",
    "min_edu_level": "最低学历",
    "job_description": "工作描述",
}

_max_seq_len_per_feature = {}
for k in _resume_col_name_mapping.values():
    if k == "工作经验":
        _max_seq_len_per_feature[k] = 256  # used in InEXIT
    else:
        _max_seq_len_per_feature[k] = 16

for k in _jd_col_name_mapping.values():
    if k == "工作描述":
        _max_seq_len_per_feature[k] = 256
    else:
        _max_seq_len_per_feature[k] = 16

_max_key_seq_length = 16  # encode the key name


def augment_aliyun_resume(
    resume: dict,
    augment_fn: Callable,
    data_type: str = 'positive'
):
    # we want to augment the resume
    augmented_resume = deepcopy(resume)
    # augment the experiences and projects section IF not UNKNOWN
    experience = augmented_resume['工作经验']
    augmented_resume['工作经验'] = augment_fn(experience, data_type=data_type)
    return augmented_resume


def augment_aliyun_jd(
    jd: dict,
    augment_fn: Callable,
    data_type: str = 'positive'
):
    augmented_jd = deepcopy(jd)
    job_description = augmented_jd['工作描述']
    augmented_jd['工作描述'] = augment_fn(job_description, data_type=data_type)
    return augmented_jd


ALIYUN_CONFIG = {
    "max_seq_len_per_feature": _max_seq_len_per_feature,
    "max_key_seq_length": _max_key_seq_length,
    "resume_taxon_token": "简历信息",
    "job_taxon_token": "职位信息",
    "resume_key_names": list(_resume_col_name_mapping.values()),
    "job_key_names": list(_jd_col_name_mapping.values()),
    "resume_aug_fn": augment_aliyun_resume,
    "job_aug_fn": augment_aliyun_jd,
}