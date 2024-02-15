from transformers import (
    AutoTokenizer, AutoModelForMaskedLM,
    DataCollatorForLanguageModeling,
    Trainer, TrainingArguments
)
from src.schema.document import Document
from tqdm.auto import tqdm
from typing import List, Dict
import pickle
import torch
import wandb
import random
import csv


class DocumentDataset(torch.utils.data.Dataset):
    def __init__(self, 
            raw_data: List[Document],
            tokenizer,
            max_length=512,
            start_data_idx=0,
            end_data_idx=None,
            shuffle=False):
        self.max_length = max_length
        self.start_data_idx = start_data_idx
        self.end_data_idx = end_data_idx
        self.tokenizer = tokenizer

        self.data = self.encode_data(raw_data)
        if shuffle:
            # usually the training data files are ALREADY shuffled
            # in the case of few shot experiments, we want to explicitly shuffle the data
            random.seed(42)
            random.shuffle(self.data)
        return
    
    def encode_into_tokenized_chunks(self, text: str):
        all_tokenized = self.tokenizer(
            text,
            truncation=False,
            return_tensors="pt"
        )
        all_tokenized = {k: v.squeeze(0) for k, v in all_tokenized.items()}
        chunked_encoded_ids = []
        chunked_attention_mask = []
        chunked_labels = []
        overlap_size = int(0.1 * self.max_length)

        for i in range(0, len(all_tokenized["input_ids"]), self.max_length - overlap_size):
            encoded_ids = all_tokenized["input_ids"][i:i+self.max_length]
            encoded_ids[0] = self.tokenizer.cls_token_id
            attention_mask = all_tokenized["attention_mask"][i:i+self.max_length]

            # pad if necessary
            if len(encoded_ids) < self.max_length:
                zeros = torch.zeros(self.max_length - len(encoded_ids), dtype=torch.long)
                padding = zeros * self.tokenizer.pad_token_id
                encoded_ids = torch.cat((encoded_ids, padding), dim=0)
                attention_mask = torch.cat((attention_mask, zeros), dim=0)
            chunked_encoded_ids.append(encoded_ids)
            chunked_attention_mask.append(attention_mask)
            chunked_labels.append(encoded_ids.clone())

        return chunked_encoded_ids, chunked_attention_mask, chunked_labels
        
    
    def encode_data(self, raw_data: List[Document]):
        encoded_data = []
        for data in tqdm(raw_data[self.start_data_idx:self.end_data_idx], desc="Encoding data"):
            title = data.metadata['type']
            formatted_data = f"{title}: {data.content}"
            encoded_ids, attention_masks, labels = self.encode_into_tokenized_chunks(formatted_data)
            
            for encoded_id, attention_mask, label in zip(encoded_ids, attention_masks, labels):
                tokenized = {
                    "input_ids": encoded_id,
                    "attention_mask": attention_mask,
                    "labels": label
                }
                encoded_data.append(tokenized)
            #	# checking
            # 	decoded = self.tokenizer.decode(encoded_id)
            # 	print(decoded)
            # 	print('\n\n')
            # print('\n\n\n\n')
        print(f"Processed {len(encoded_data)} documents")
        return encoded_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
    

def get_dataset(tokenizer):
    random.seed(0)
    # all data
    resume_data_path = "dataset/train/resume_documents.pkl"
    jd_data_path = "dataset/train/jd_documents.pkl"

    with open(resume_data_path, "rb") as fread:
        resume_docs = pickle.load(fread)
    with open(jd_data_path, "rb") as fread:
        jd_docs = pickle.load(fread)

    # eval_jd_id
    with open("dataset/eval/eval_jd_ids.csv", "r") as fread:
        csvreader = csv.reader(fread, delimiter='\n')
        eval_jd_ids = set([int(x[0]) for x in csvreader])
        print(f"Found {len(eval_jd_ids)} eval jd ids")
    with open("dataset/eval/eval_talent_ids.csv", "r") as fread:
        csvreader = csv.reader(fread, delimiter='\n')
        eval_talent_ids = set([int(x[0]) for x in csvreader])
        print(f"Found {len(eval_talent_ids)} eval talent ids")
    # test_jd_id
    with open("dataset/test/test_jd_ids.csv", "r") as fread:
        csvreader = csv.reader(fread, delimiter='\n')
        test_jd_ids = set([int(x[0]) for x in csvreader])
        print(f"Found {len(test_jd_ids)} test jd ids")
    with open("dataset/test/test_talent_ids.csv", "r") as fread:
        csvreader = csv.reader(fread, delimiter='\n')
        test_talent_ids = set([int(x[0]) for x in csvreader])
        print(f"Found {len(test_talent_ids)} test talent ids")
    
    all_docs = resume_docs + jd_docs
    train_docs = []
    eval_docs = []
    for doc in all_docs:
        if doc.metadata['type'] == 'resume':
            if doc.metadata['talent_id'] in eval_talent_ids:
                eval_docs.append(doc)
            elif doc.metadata['talent_id'] in test_talent_ids:
                continue
            else:
                train_docs.append(doc)
        elif doc.metadata['type'] == 'jd':
            if doc.metadata['jd_id'] in eval_jd_ids:
                eval_docs.append(doc)
            elif doc.metadata['jd_id'] in test_jd_ids:
                continue
            else:
                train_docs.append(doc)
        else:
            raise ValueError("Unknown document type")
    print(f"Processed {len(train_docs)} train docs out of {len(all_docs)} docs")
    print(f"Processed {len(eval_docs)} eval docs out of {len(all_docs)} docs")
    
    random.shuffle(train_docs)

    train_dset = DocumentDataset(
        raw_data=train_docs,
        tokenizer=tokenizer,
        end_data_idx=None,
        shuffle=False
    )
    eval_dset = DocumentDataset(
        raw_data=eval_docs,
        tokenizer=tokenizer,
        end_data_idx=None,
        shuffle=False
    )
    return train_dset, eval_dset


if __name__ == "__main__":
    # processed by python init_db.py --db_cfg short --source_config configs/databricks_sources_official.json
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name", type=str, default="bert-base-multilingual-cased"
    )
    parser.add_argument(
        '--output_dir', type=str, default="model_checkpoints/bert/base-multilingual-cased_mlm"
    )
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    train_dset, eval_dset = get_dataset(tokenizer)

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm_probability=0.15
    )

    model = AutoModelForMaskedLM.from_pretrained(args.model_name)
    
    training_args = TrainingArguments(
        output_dir = args.output_dir,
        learning_rate = 1e-5,
        per_device_train_batch_size = 16,
        warmup_ratio = 0.1,
        lr_scheduler_type = "cosine",
        num_train_epochs = 5,
        weight_decay = 0.01,
        logging_steps = 100,
        evaluation_strategy = "steps",
        eval_steps = 2000,
        metric_for_best_model = "eval_loss",
        save_strategy = "steps",
        save_steps = 2000,
        save_total_limit = 1,
        report_to = "wandb",
        push_to_hub=False,
    )

    if 'wandb' in training_args.report_to:
        run = wandb.init(
            project='resume',
            entity='tamarin',
            name=training_args.output_dir.split("/")[-1] or None,
            group='resume_mlm',
            config=training_args.to_dict(),
        )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dset,
        eval_dataset=eval_dset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    trainer.train()
    