from dataclasses import dataclass
from src.preprocess.eda import eda
from src.preprocess.eda_zh import eda as eda_zh
import re
import random


@dataclass
class EDAConfig:
    """controls how data augmentation is done
    """
    alpha_sr: float = 0.1
    alpha_ri: float = 0.1
    alpha_rs: float = 0.1
    alpha_rd: float = 0.1
    num_aug: int = 2
    seed: int = 42
    ## newly added
    row_shuffle_prob: float = 0.1
    row_del_p: float = 0.1
    paragraph_del_p: float = 0.1


def is_chinese(text: str):
    chinese_sections = re.findall(r'[\u4e00-\u9fff]+', text)
    all_chinese = ''.join(chinese_sections)
    if len(all_chinese) > 0.2 * len(text):
        return True
    return False


def _eda_en_augment_paragraph(paragraph: str, config: EDAConfig):
    augmented_lines = []
    for line in paragraph.split('\n'):
        # too short, or UNKNOWN!
        if len(line.strip()) < 30:
            augmented_lines.append(line)
            continue
        try:
            aug_line = eda(
                line,
                alpha_sr=config.alpha_sr,
                alpha_ri=config.alpha_ri,
                alpha_rs=config.alpha_rs,
                p_rd=config.alpha_rd,
                num_aug=config.num_aug,
            )[1].strip()
        except Exception as _:
            aug_line = line
        augmented_lines.append(aug_line)
    augmented_paragraph = "\n".join(augmented_lines)
    return augmented_paragraph

def _eda_zh_augment_paragraph(paragraph: str, config: EDAConfig):
    augmented_lines = []
    for line in paragraph.split('\n'):
        # too short, or UNKNOWN!
        if len(line.strip()) < 10:
            augmented_lines.append(line)
            continue
        try:
            aug_line = eda_zh(
                line,
                alpha_sr=config.alpha_sr,
                alpha_ri=config.alpha_ri,
                alpha_rs=config.alpha_rs,
                p_rd=config.alpha_rd,
                num_aug=config.num_aug,
            )[1].strip()
        except Exception as _:
            aug_line = line
        augmented_lines.append(aug_line)
    augmented_paragraph = "\n".join(augmented_lines)
    return augmented_paragraph


def _eda_augment_paragraph(paragraph: str, config: EDAConfig):
    if is_chinese(paragraph):
        return _eda_zh_augment_paragraph(paragraph, config)
    else:
        return _eda_en_augment_paragraph(paragraph, config)


def eda_augment_paragraph_w_type(
    text: str,
    config: EDAConfig,
    data_type: str
):
    assert data_type in ['positive', 'negative', 'noop', 'pretrain']
    # e.g. used on validation set
    if data_type == 'noop':
        return text
    
    augmented_paragraph = _eda_augment_paragraph(text, config)
    if data_type == 'negative' or data_type == 'pretrain':
        if random.random() < config.paragraph_del_p:
            return 'UNKNOWN'
        
        # additional augmentation such as row deletion and row swapping
        aug_lines = augmented_paragraph.split('\n')
        more_augmentation_lines = []
        # the first line could be "title information" such as "Experience:"
        for line in aug_lines[1:]:
            if random.random() < config.row_del_p:
                continue
            more_augmentation_lines.append(line)
        if random.random() < config.row_shuffle_prob:
            random.shuffle(more_augmentation_lines)
        final_aug_lines = [aug_lines[0]] + more_augmentation_lines
        augmented_paragraph = "\n".join(final_aug_lines)
    return augmented_paragraph