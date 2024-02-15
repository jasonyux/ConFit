from typing import Callable
from src.preprocess.flatten import flatten_dict


_max_seq_len_per_resume_feature = {
    "education": 256,
    "experiences": 512,
    "location": 64,
    "preferred_locations": 64,
    "industry": 512,
    "languages": 64,
    "skills": 256,
    "projects": 512,
}

_max_seq_len_per_job_feature = {
    'Job Title': 64,
    'Job Description/Responsibilities': 512,
    'Job Location': 64,
    'Job Position Type': 64,
    'Required Qualifications/Skills': 256,
    'Preferred Qualifications/Skills': 256,
    'Company Name': 64,
    'Company Description': 128,
    'Company Location': 64,
}


_max_key_seq_length = 16  # encode the key name


def flatten_proprietary_resume(resume: dict):
    flattened_resume = {}
    for k, v in resume.items():
        if k == 'user_id':
            flattened_resume[k] = v
        else:
            if isinstance(v, list):
                entries = []
                for vv in v:
                    entries.append(flatten_dict(vv))
                flattened_resume[k] = '\n\n'.join(entries)
            else:
                flattened_resume[k] = flatten_dict(v)
    return flattened_resume


def augment_proprietary_resume(
    resume: dict,
    augment_fn: Callable,
    data_type: str = 'positive'
) -> dict:
    """augment the resume by EDAing its experience and projects

    Args:
        resume (dict): _description_
        augment_fn (Callable): _description_
        data_type (str, optional): if this resume is used as a positive sample or not. Defaults to 'positive'.

    Returns:
        _type_: _description_
    """
    assert data_type in ['positive', 'negative', 'noop', 'pretrain']
    augmented_resume = {}
    for k, v in resume.items():
        if k == 'user_id':
            augmented_resume[k] = v
        else:
            augmented_resume[k] = eval(v)
    
    # augment the experiences and projects section IF not UNKNOWN
    experiences = augmented_resume['experiences']
    for experience in experiences:
        description = experience['description']
        augmented_description = augment_fn(description, data_type=data_type)
        experience['description'] = augmented_description
    
    projects = augmented_resume['projects']
    for project in projects:
        description = project['description']
        augmented_description = augment_fn(description, data_type=data_type)
        project['description'] = augmented_description

    if data_type == 'pretrain':
        # augment everything
        experiences = augmented_resume['experiences']
        for experience in experiences:
            title = experience['title']
            experience['title'] = augment_fn(title, data_type=data_type)
            start_date = experience['start_date']
            experience['start_date'] = augment_fn(start_date, data_type=data_type)
            end_date = experience['end_date']
            experience['end_date'] = augment_fn(end_date, data_type=data_type)

        projects = augmented_resume['projects']
        for project in projects:
            name = project['project_name']
            project['project_name'] = augment_fn(name, data_type=data_type)
            title = project['title']
            project['title'] = augment_fn(title, data_type=data_type)
            start_date = project['start_date']
            project['start_date'] = augment_fn(start_date, data_type=data_type)
            end_date = project['end_date']
            project['end_date'] = augment_fn(end_date, data_type=data_type)
        
        location = augmented_resume['location']
        province = location['official_province']
        location['official_province'] = augment_fn(province, data_type=data_type)
        city = location['official_city']
        location['official_city'] = augment_fn(city, data_type=data_type)

        educations = augmented_resume['education']
        for education in educations:
            major_name = education['major_name']
            augmented_major_name = augment_fn(major_name, data_type=data_type)
            education['major_name'] = augmented_major_name
        
        industries = augmented_resume['industry']
        for industry in industries:
            industry_name = industry['name']
            augmented_industry_name = augment_fn(industry_name, data_type=data_type)
            industry['name'] = augmented_industry_name
        
        skills = augmented_resume['skills']
        for skill in skills:
            skill_name = skill['skill_name']
            augmented_skill_name = augment_fn(skill_name, data_type=data_type)
            skill['skill_name'] = augmented_skill_name
    
    # flatten the resume
    augmented_resume = flatten_proprietary_resume(augmented_resume)
    return augmented_resume


def augment_proprietary_jd(
    jd: dict,
    augment_fn: Callable,
    data_type: str = 'positive'
):
    assert data_type in ['positive', 'negative', 'noop', 'pretrain']

    keys_to_augment = [
        "Company Description",
        "Job Description/Responsibilities",
        "Required Qualifications/Skills",
        "Preferred Qualifications/Skills",
    ]
    if data_type == "pretrain":
        keys_to_augment += [
            "Job Title",
            "Company Location",
            "Company Description"
        ]
    augmented_jd = {}
    for k, v in jd.items():
        if k == 'jd_no':
            augmented_jd[k] = v
            continue
        if k in keys_to_augment:
            augmented_v = augment_fn(v, data_type=data_type)
            augmented_jd[k] = augmented_v
        else:
            augmented_jd[k] = v
    return augmented_jd


PROPRIETARY_CONFIG = {
    "max_seq_len_per_feature": {
        **_max_seq_len_per_resume_feature,
        **_max_seq_len_per_job_feature,
    },
    "max_key_seq_length": _max_key_seq_length,
    "resume_taxon_token": "Resume Content",
    "job_taxon_token": "Job Information",
    "resume_key_names": list(_max_seq_len_per_resume_feature.keys()),
    "job_key_names": list(_max_seq_len_per_job_feature.keys()),
    "resume_aug_fn": augment_proprietary_resume,
    "job_aug_fn": augment_proprietary_jd,
}