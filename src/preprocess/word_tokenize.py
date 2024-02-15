from nltk.corpus import stopwords
import nltk
import string
import jieba
import numpy as np
import regex
import ast


def word_tokenize_chinese(text):
    chinese_punc = string.punctuation+'：；，。？、~@#￥%……&*《》（）'
    stop_words = set(stopwords.words("chinese"))
    word_tokens = jieba.lcut(text)
    filtered_text = [word.lower() for word in word_tokens if word not in stop_words and word.strip() not in chinese_punc]
    return filtered_text


def word_tokenize_english(text):
    stop_words = set(stopwords.words("english"))
    word_tokens = nltk.word_tokenize(text)
    filtered_text = [word.lower() for word in word_tokens if word not in stop_words and word.strip() not in string.punctuation]
    return filtered_text


def detect_language_and_word_tokenize(text):
    # detect language
    # if chinese, use jieba
    # if english, use nltk
    chinese_words = regex.findall(r'\p{Han}+', text)
    chinese_length = len(''.join(chinese_words))
    # average length of enligsh word is 4.7
    if chinese_length > 0.15 * len(text):
        return word_tokenize_chinese(text)
    else:
        return word_tokenize_english(text)


def ali_resume_to_words_simple(resume_data: dict):
    output_words = []
    for f, value in resume_data.items():
        if f in ['user_id']:
            continue
        if isinstance(value, float) and np.isnan(value):
            continue
        elif isinstance(value, (float, int)):
            tokenized_values = [f'{value}']
        else:
            tokenized_values = word_tokenize_chinese(value)
        output_words.extend(tokenized_values)
    return output_words


def ali_resume_to_words(resume_data: dict):
    output_words = []
    for f, value in resume_data.items():
        if f in ['user_id']:
            continue
        if isinstance(value, float) and np.isnan(value):
            continue
        elif f in ['experience']:
            # value is tetx that needs to be tokenized
            tokenized_values = word_tokenize_chinese(value)
        # other ones
        elif f in ['desire_jd_industry_id', 'desire_jd_type_id', 'cur_industry_id', 'cur_jd_type']:
            tokenized_values = value.split('/')
            tokenized_values = [f'{f}_{v}' for v in tokenized_values]
        elif f in ['desire_jd_city_id']:
            tokenized_values = value.split(',')
            tokenized_values = [f'desired_city_{v}' for v in tokenized_values if v != '-']
        elif f in ['live_city_id', 'desire_jd_salary_id', 'cur_salary_id', 'cur_degree_id', 'birthday', 'start_work_date']:
            # no-op as they are categorical to begin with
            tokenized_values = [f'{f}_{value}']
        else:
            raise ValueError(f'unknown field {f}')
        output_words.extend(tokenized_values)
    return output_words


def ali_jd_to_words_simple(jd_data: dict):
    """tokenize a job into words, by just tokenizing the words agnostic of the field name
    """
    output_words = []
    for f, value in jd_data.items():
        if f in ['jd_no']:
            continue
        if isinstance(value, float) and np.isnan(value):
            continue
        elif isinstance(value, (float, int)):
            tokenized_values = [f'{value}']
        else:
            tokenized_values = word_tokenize_chinese(value)
        output_words.extend(tokenized_values)
    return output_words


def ali_jd_to_words(jd_data: dict):
    output_words = []
    for f, value in jd_data.items():
        if f in ['jd_no']:
            continue
        if isinstance(value, float) and np.isnan(value):
            continue
        elif f in ['job_description']:
            tokenized_values = word_tokenize_chinese(value)
        elif f in ['jd_title']:
            tokenized_values = word_tokenize_chinese(value)
            tokenized_values = [f'{f}_{v}' for v in tokenized_values]
        elif f in ['jd_sub_type']:
            tokenized_values = value.split('/')
            tokenized_values = [f'{f}_{v}' for v in tokenized_values]
        elif f in ['city', 'require_nums', 'start_date', 'end_date', 'is_travel', 'min_years', 'min_edu_level']:
            # no-op
            tokenized_values = [f'{f}_{value}']
        elif f in ['salary']:
            # special processing
            min_salary, max_salary = value.split('-')
            tokenized_values = [f'salary_{min_salary}', f'salary_{max_salary}']
        else:
            raise ValueError(f'unknown field {f}')
        output_words.extend(tokenized_values)
    return output_words


def __tokenize_dict_in_proprietary_resume(data: str, type: str, field_name: str):
    # for a few fields such as experience or project, we may want to tokenize WITHOUT prefix
    # note: content may be english or chinese
    if data == 'UNKNOWN':
        return []  # we do not want bow or tf-idf to even consider this
    
    full_prefix = f'{type}_{field_name}'
    if type == 'education':
        if field_name in ['start_date', 'end_date', 'degree', 'college_name', 'qs2021_ranking']:
            # no-op
            tokenized_values = [f'{full_prefix}_{data}']
        elif field_name in ['major_name']:
            tokenized_values = detect_language_and_word_tokenize(data)
            tokenized_values = [f'{full_prefix}_{v}' for v in tokenized_values]
        else:
            raise ValueError(f'unknown field {field_name} from {type}')
    elif type == 'experiences':
        if field_name in ['start_date', 'end_date', 'company_name', 'fortune2020_ranking']:
            # no-op
            tokenized_values = [f'{full_prefix}_{data}']
        elif field_name in ['title', 'location', 'description']:
            tokenized_values = detect_language_and_word_tokenize(data)
            tokenized_values = [f'{full_prefix}_{v}' for v in tokenized_values]
        else:
            raise ValueError(f'unknown field {field_name} from {type}')
    elif type in ['location', 'preferred_locations']:
        if field_name in ['official_city', 'official_province', 'official_country']:
            # no-op
            tokenized_values = [f'{full_prefix}_{data}']
        elif field_name in ['location']:
            tokenized_values = detect_language_and_word_tokenize(data)
            tokenized_values = [f'{full_prefix}_{v}' for v in tokenized_values]
        else:
            raise ValueError(f'unknown field {field_name} from {type}')
    elif type == 'industry':
        if field_name in ['name']:
            tokenized_values = data.split('.')
            tokenized_values = [f'{full_prefix}_{v}' for v in tokenized_values]
        else:
            raise ValueError(f'unknown field {field_name} from {type}')
    elif type == 'languages':
        if field_name in ['name']:
            # no-op
            tokenized_values = [f'{full_prefix}_{data}']
        else:
            raise ValueError(f'unknown field {field_name} from {type}')
    elif type == 'skills':
        if field_name in ['skill_name']:
            # no-op
            tokenized_values = [f'{full_prefix}_{data}']
        else:
            raise ValueError(f'unknown field {field_name} from {type}')
    elif type == 'projects':
        if field_name in ['start_date', 'end_date']:
            # no-op
            tokenized_values = [f'{full_prefix}_{data}']
        elif field_name in ['project_name', 'title', 'description']:
            tokenized_values = detect_language_and_word_tokenize(data)
            tokenized_values = [f'{full_prefix}_{v}' for v in tokenized_values]
        else:
            raise ValueError(f'unknown field {field_name} from {type}')
    else:
        raise ValueError(f'unknown field {type}')
    return tokenized_values


def proprietary_resume_to_words_simple(resume_data: dict):
    """tokenize a resume into words, by just tokenizing the words agnostic of the field name
    """
    output_words = []
    for f, value in resume_data.items():
        if f in ['user_id']:
            continue

        if value.startswith('[') or value.startswith('{'):
            value = ast.literal_eval(value)

        # each value is either a list of a dict
        if isinstance(value, list):
            for v in value:
                # a dict
                for kk, vv in v.items():
                    # the actual content
                    tokenized_values = detect_language_and_word_tokenize(vv)
                    output_words.extend(tokenized_values)
        elif isinstance(value, dict):
            for kk, vv in value.items():
                # the actual content
                tokenized_values = detect_language_and_word_tokenize(vv)
                output_words.extend(tokenized_values)
    return output_words


def proprietary_resume_to_words(resume_data: dict):
    """tokenize a resume into words, but words are now field dependent
    """
    output_words = []
    for f, value in resume_data.items():
        if f in ['user_id']:
            continue

        if value.startswith('[') or value.startswith('{'):
            value = ast.literal_eval(value)

        # each value is either a list of a dict
        if isinstance(value, list):
            for v in value:
                # a dict
                for kk, vv in v.items():
                    # the actual content
                    tokenized_values = __tokenize_dict_in_proprietary_resume(vv, f, kk)
                    output_words.extend(tokenized_values)
        elif isinstance(value, dict):
            for kk, vv in value.items():
                # the actual content
                tokenized_values = __tokenize_dict_in_proprietary_resume(vv, f, kk)
                output_words.extend(tokenized_values)
    return output_words


def proprietary_jd_to_words_simple(jd_data: dict):
    """tokenize a job into words, by just tokenizing the words agnostic of the field name
    """
    output_words = []
    for f, value in jd_data.items():
        if f in ['jd_no']:
            continue

        if value == 'UNKNOWN':
            tokenized_values = []
        else:
            tokenized_values = detect_language_and_word_tokenize(value)
        output_words.extend(tokenized_values)
    return output_words


def proprietary_jd_to_words(jd_data: dict):
    """tokenize a job into words, but words are now field dependent
    """
    output_words = []
    for f, value in jd_data.items():
        if f in ['jd_no']:
            continue

        if value == 'UNKNOWN':
            tokenized_values = []
        elif f in ['company_name']:
            # no-op
            tokenized_values = [f'{f}_{value}']
        else:
            tokenized_values = detect_language_and_word_tokenize(value)
            tokenized_values = [f'{f}_{v}' for v in tokenized_values]
        output_words.extend(tokenized_values)
    return output_words