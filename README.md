# ConFit

This repository contains an official implementation of ConFit, which is described in the *RecSys 24'* paper:

**ConFit: Improving Resume-Job Matching using Data Augmentation and Contrastive Learning**<br>
*Xiao Yu, Jinzhong Zhang, Zhou Yu*


## Dependencies

1. install the required packages with `pip install -r requirements.txt`
2. make sure you have a `wandb` account and have logged in with `wandb login` (if not you need to remove the `wandb` related code in the scripts)
3. run `export PYTHONPATH=$(pwd)` before running any scripts in this repository

## Data

All of our training data is placed under `dataset/` folder. For privacy concerns, we replaced all the contents to placeholder such as `[[LONG_ENGLISH_TEXT]]`.
- If you would like to use our training script/model **with your own data**, you can simply follow the same format and replace the placeholder with your own data.
- If you would like to **have access to our data**, please contact us directly.

However, we note that **our dummy data should be compatible** with our provided scripts, since we only modified the *content*. So to test if you have set up everything correctly, you can directly try some of our examples in the [Training](#Training) section.

## Training

### ConFit Training

Train ConFIT with BERT on the IntelliPro dataset
```bash
python runners/trainer/train_confit.py \
--save_path model_checkpoints/confit/IntelliPro_bert \
--resume_data_path dataset/IntelliPro/all_resume_w_flattened_text_eda_n_chatgpt.csv \
--job_data_path dataset/IntelliPro/all_jd_eda_n_chatgpt.csv \
--train_label_path dataset/IntelliPro/train_labeled_data_eda_n_chatgpt.jsonl \
--num_resume_features 8 --num_job_features 9 \
--valid_label_path dataset/IntelliPro/valid_classification_data.jsonl \
--classification_data_path dataset/IntelliPro/test_classification_data.jsonl \
--rank_resume_data_path dataset/IntelliPro/rank_resume.json \
--rank_job_data_path dataset/IntelliPro/rank_job.json \
--dataset_type IntelliPro \
--train_batch_size 8 \
--num_hard_negatives 8 \
--val_batch_size 8 \
--gradient_accumulation_steps 2 \
--weight_decay 1e-2 \
--num_suit_encoder_layers 0 \
--finegrained_loss noop \
--pretrained_encoder bert-base-multilingual-cased \
--temperature 1.0 \
--log_group IntelliPro/confit \
--do_both_rj_hard_neg true
```

Train ConFIT with E5-large on the IntelliPro dataset
```bash
python runners/trainer/train_confit.py \
--save_path model_checkpoints/confit/IntelliPro_e5 \
--resume_data_path dataset/IntelliPro/all_resume_w_flattened_text_eda_new_n_chatgpt.csv \
--job_data_path dataset/IntelliPro/all_jd_eda_new_n_chatgpt.csv \
--train_label_path dataset/IntelliPro/train_labeled_data_eda_new_n_chatgpt.jsonl \
--num_resume_features 8 --num_job_features 9 \
--valid_label_path dataset/IntelliPro/valid_classification_data.jsonl \
--classification_data_path dataset/IntelliPro/test_classification_data.jsonl \
--rank_resume_data_path dataset/IntelliPro/rank_resume.json \
--rank_job_data_path dataset/IntelliPro/rank_job.json \
--dataset_type IntelliPro \
--train_batch_size 8 \
--num_hard_negatives 4 \
--val_batch_size 8 \
--gradient_accumulation_steps 2 \
--weight_decay 1e-2 \
--finegrained_loss noop \
--num_suit_encoder_layers 0 \
--pretrained_encoder intfloat/multilingual-e5-large \
--word_emb_dim 1024 \
--hidden_size 1024 \
--query_prefix query \
--temperature 1.0 \
--log_group IntelliPro/confit \
--do_both_rj_hard_neg true \
--strategy deepspeed_stage_2 \
--precision bf16
```

Train ConFIT with BERT on the AliYun dataset
```bash
runners/trainer/train_confit.py \
--save_path model_checkpoints/confit/aliyun_bert \
--resume_data_path dataset/AliTianChi/all_resume_w_updated_colnames_eda_n_chatgpt.csv \
--job_data_path dataset/AliTianChi/all_job_w_updated_colnames_eda_n_chatgpt.csv \
--train_label_path dataset/AliTianChi/train_labeled_data_eda_n_chatgpt.jsonl \
--train_batch_size 8 \
--num_hard_negatives 8 \
--val_batch_size 8 \
--gradient_accumulation_steps 2 \
--weight_decay 1e-2 \
--finegrained_loss noop \
--num_suit_encoder_layers 0 \
--pretrained_encoder bert-base-chinese \
--temperature 1.0 \
--log_group aliyun/confit \
--do_both_rj_hard_neg true
```

Train ConFIT with E5-large on the AliYun dataset
```bash
python runners/trainer/train_confit.py \
--save_path model_checkpoints/confit/aliyun_e5 \
--resume_data_path dataset/AliTianChi/all_resume_w_updated_colnames_eda_n_chatgpt.csv \
--job_data_path dataset/AliTianChi/all_job_w_updated_colnames_eda_n_chatgpt.csv \
--train_label_path dataset/AliTianChi/train_labeled_data_eda_n_chatgpt.jsonl \
--train_batch_size 8 \
--num_hard_negatives 8 \
--val_batch_size 8 \
--gradient_accumulation_steps 2 \
--weight_decay 1e-2 \
--num_suit_encoder_layers 0 \
--finegrained_loss noop \
--pretrained_encoder intfloat/multilingual-e5-large \
--word_emb_dim 1024 \
--hidden_size 1024 \
--query_prefix query \
--temperature 1.0 \
--log_group aliyun/confit \
--do_both_rj_hard_neg true
```

### Training Baselines

Since there are a large combination of models and training configurations, please refer to our paper for more details. We highlight one example for each baseline here.


Train **InEXIT** with BERT on the IntelliPro data:
```bash
python runners/trainer/train_inexit.py \
--save_path model_checkpoints/InEXIT/IntelliPro_bce \
--resume_data_path dataset/IntelliPro/all_resume_w_flattened_text.csv \
--job_data_path dataset/IntelliPro/all_jd.csv \
--train_label_path dataset/IntelliPro/train_labeled_data.jsonl \
--valid_label_path dataset/IntelliPro/valid_classification_data.jsonl \
--classification_data_path dataset/IntelliPro/test_classification_data.jsonl \
--rank_resume_data_path dataset/IntelliPro/rank_resume.json \
--rank_job_data_path dataset/IntelliPro/rank_job.json \
--dataset_type IntelliPro \
--train_batch_size 8 --val_batch_size 8 \
--pretrained_encoder bert-base-multilingual-cased \
--gradient_accumulation_steps 2 \
--weight_decay 1e-2 \
--log_group IntelliPro
```

Train **DPGNN** with BERT on the IntelliPro data:
```bash
python runners/trainer/train_dpgnn.py \
--save_path model_checkpoints/DPGNN/IntelliPro_bert_normed \
--resume_data_path dataset/IntelliPro/all_resume_w_flattened_text.csv \
--job_data_path dataset/IntelliPro/all_jd.csv \
--train_label_path dataset/IntelliPro/train_labeled_data.jsonl \
--num_resume_features 8 --num_job_features 9 \
--valid_label_path dataset/IntelliPro/valid_classification_data.jsonl \
--classification_data_path dataset/IntelliPro/test_classification_data.jsonl \
--rank_resume_data_path dataset/IntelliPro/rank_resume.json \
--rank_job_data_path dataset/IntelliPro/rank_job.json \
--dataset_type IntelliPro \
--bpr_loss_version original \
--train_batch_size 8 \
--val_batch_size 8 \
--gradient_accumulation_steps 2 \
--weight_decay 1e-2 \
--log_group IntelliPro/dpgnn \
--pretrained_encoder bert-base-chinese \
--do_normalize true
```

Train **MV-CoN** with BERT on the AliYun data:
```bash
python runners/trainer/train_mvcon.py \
--save_path model_checkpoints/MV-CoN/IntelliPro \
--resume_data_path dataset/IntelliPro/all_resume_w_flattened_text.csv \
--job_data_path dataset/IntelliPro/all_jd.csv \
--train_label_path dataset/IntelliPro/train_labeled_data.jsonl \
--valid_label_path dataset/IntelliPro/valid_classification_data.jsonl \
--classification_data_path dataset/IntelliPro/test_classification_data.jsonl \
--rank_resume_data_path dataset/IntelliPro/rank_resume.json \
--rank_job_data_path dataset/IntelliPro/rank_job.json \
--dataset_type IntelliPro \
--num_resume_features 8 --num_job_features 9 \
--train_batch_size 4 --val_batch_size 4 \
--pretrained_encoder bert-base-multilingual-cased \
--gradient_accumulation_steps 4 \
--weight_decay 1e-2 \
--log_group IntelliPro
```

## Evaluation

There are many models and baselines we can evaluate. This not only includes evaluating the neural models we trained in the previous section, but also evaluating off-the-shelf embeddings and baselines such as BM25.


Evaluating XGBoost, BM25, and off-the-shelf embeddings:

- evaluate xgboost with tfidf/bow. For example:
  ```bash
  python runners/tester/test_xgboost.py --dset aliyun --encoder tfidf
  ```
- evaluate xgboost with embedding vectors. For example:
  ```bash
  python runners/tester/test_xgboost_w_embedding.py \
  --resume_train_valid_embedding_path model_checkpoints/openai/aliyun/all_resume_train_embeddings.jsonl \
  --resume_test_embedding_path model_checkpoints/openai/aliyun/all_resume_test_embeddings.jsonl \
  --job_train_valid_embedding_path model_checkpoints/openai/aliyun/all_job_train_embeddings.jsonl \
  --job_test_embedding_path model_checkpoints/openai/aliyun/all_job_test_embeddings.jsonl \
  --dset aliyun
  ```
- evaluate off-the-shelf embeddings. For example:
  ```bash
  python runners/tester/test_raw_embedding.py \
  --dset aliyun \
  --embedding_folder model_checkpoints/xlm-roberta/aliyun
  ```
- evaluate bm25. For example:
  ```bash
  python runners/tester/test_bm25.py --dset aliyun
  ```

Evaluating neural models we trained before

- evaluate ConFit:
  ```bash
  python runners/tester/test_confit.py \
  --model_path model_checkpoints/confit/IntelliPro_bert \
  --model_checkpoint_name epoch=2-step=975.ckpt \
  --resume_data_path dataset/IntelliPro/all_resume_w_flattened_text.csv \
  --job_data_path dataset/IntelliPro/all_jd.csv \
  --classification_validation_data_path dataset/IntelliPro/valid_classification_data.jsonl \
  --classification_data_path dataset/IntelliPro/test_classification_data.jsonl \
  --rank_resume_data_path dataset/IntelliPro/rank_resume.json \
  --rank_job_data_path dataset/IntelliPro/rank_job.json \
  --dataset_type IntelliPro \
  --wandb_run_id rcskbb25  # if you want to log the results to a wandb run
  ```
- evaluate InEXIT on the IntelliPro dataset:
  ```bash
  python runners/tester/test_inexit.py \
  --model_path model_checkpoints/InEXIT/IntelliPro_bce \
  --model_checkpoint epoch\=1-step\=1644.ckpt \
  --resume_data_path dataset/IntelliPro/all_resume_w_flattened_text.csv \
  --job_data_path dataset/IntelliPro/all_jd.csv \
  --classification_validation_data_path dataset/IntelliPro/valid_classification_data.jsonl \
  --classification_data_path dataset/IntelliPro/test_classification_data.jsonl \
  --rank_resume_data_path dataset/IntelliPro/rank_resume.json \
  --rank_job_data_path dataset/IntelliPro/rank_job.json \
  --dataset_type IntelliPro \
  --wandb_run_id uytnmv1q  # if you want to log the results to a wandb run
  ```
- evaluate DPGNN on the IntelliPro dataset:
  ```bash
  python runners/tester/test_dpgnn.py \
  --model_path model_checkpoints/DPGNN/IntelliPro_bert_normed \
  --model_checkpoint epoch=9-step=780.ckpt \
  --resume_data_path dataset/IntelliPro/all_resume_w_flattened_text.csv \
  --job_data_path dataset/IntelliPro/all_jd.csv \
  --classification_validation_data_path dataset/IntelliPro/valid_classification_data.jsonl \
  --classification_data_path dataset/IntelliPro/test_classification_data.jsonl \
  --rank_resume_data_path dataset/IntelliPro/rank_resume.json \
  --rank_job_data_path dataset/IntelliPro/rank_job.json \
  --dataset_type IntelliPro
  ```
- evaluate MV-CoN on the IntelliPro dataset:
  ```bash
  python runners/tester/test_mvcon.py \
  --model_path model_checkpoints/MV-CoN/IntelliPro \
  --model_checkpoint epoch=7-step=22112.ckpt \
  --resume_data_path dataset/IntelliPro/all_resume_w_flattened_text.csv \
  --job_data_path dataset/IntelliPro/all_jd.csv \
  --classification_validation_data_path dataset/IntelliPro/valid_classification_data.jsonl \
  --classification_data_path dataset/IntelliPro/test_classification_data.jsonl \
  --rank_resume_data_path dataset/IntelliPro/rank_resume.json \
  --rank_job_data_path dataset/IntelliPro/rank_job.json \
  --dataset_type IntelliPro
  ```

## Ablation Studies

Here we provide some scripted used to perform ablation studies. For a complete list of ablation studies, please refer to our paper.

 For example, to do ConFit-like training with XGBoost on the IntelliPro dataset:
```bash
python runners/tester/test_xgboost_w_contrastive.py  \
--dset IntelliPro --encoder tfidf \
--n_random_neg_per_sample 7 \
--n_hard_neg_per_sample 1
```
