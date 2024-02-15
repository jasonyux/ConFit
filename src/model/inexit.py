from torchmetrics import Precision, Recall, F1Score
from transformers import AutoModel, BatchEncoding
from transformers.utils import ModelOutput
from transformers import get_linear_schedule_with_warmup
from typing import Dict
from dataclasses import dataclass, field
from src.model.base import BaseModel
import torch
import torch.nn as nn


@dataclass
class InEXITModelArguments:
    lr: float = field(default=5e-6)  # we have a scheduler
    gradient_accumulation_steps: int = field(default=1)
    weight_decay: float = field(default=1e-9)  # 1e-9 used by InEXIT
    num_warmup_steps: float = field(default=0.1)
    word_emb_dim: int = field(default=768)
    hidden_size: int = field(default=768)
    num_heads: int = field(default=8)
    num_encoder_layers: int = field(
        default=1, 
        metadata={"help": "Number of transformer layers for self-attention"}
    )
    dropout: float = field(default=0.1)
    pretrained_encoder: str = field(default='bert-base-chinese')


class InEXITModel(BaseModel):
    """re-implementation of the InEXIT model to make it generalize to other datasets and compatible with different encoder backbones
    
    original source code: https://github.com/TaihuaShao/InEXIT/tree/main
    """
    def __init__(self, args: InEXITModelArguments):
        super(InEXITModel, self).__init__()
        self.automatic_optimization = False  # we manually do optimizer.step() and scheduler.step()
        self.args = args
        self.num_classes = 2
        self.bert = AutoModel.from_pretrained(args.pretrained_encoder)
      
        for param in self.bert.parameters():
            param.requires_grad = True
        
        self.internal_encoder = torch.nn.TransformerEncoder(
            torch.nn.TransformerEncoderLayer(
                d_model=args.word_emb_dim,
                nhead=args.num_heads,
                dim_feedforward=args.hidden_size,
                dropout=args.dropout,
                batch_first=True,
                norm_first=False,
            ),
            num_layers=args.num_encoder_layers
        )

        self.external_encoder = torch.nn.TransformerEncoder(
            torch.nn.TransformerEncoderLayer(
                d_model=args.word_emb_dim,
                nhead=args.num_heads,
                dim_feedforward=args.hidden_size,
                dropout=args.dropout,
                batch_first=True,
                norm_first=False,
            ),
            num_layers=args.num_encoder_layers
        )

        self.geek_pool = nn.AdaptiveAvgPool2d((1, args.word_emb_dim))
        self.job_pool = nn.AdaptiveAvgPool2d((1, args.word_emb_dim))

        self.mlp = nn.Sequential(
            nn.Linear(args.word_emb_dim * 3, args.word_emb_dim * 3),
            nn.ReLU(),
            nn.Linear(args.word_emb_dim * 3, 1)  # binary classification
        )

        self.metrics = {
            'train': {
                'f1': F1Score(task='multiclass', num_classes=self.num_classes, average='weighted'),
                'precision': Precision(task='multiclass', num_classes=self.num_classes, average=None),
                'recall': Recall(task='multiclass', num_classes=self.num_classes, average=None),
            },
            'val': {
                'f1': F1Score(task='multiclass', num_classes=self.num_classes, average='weighted'),
                'precision': Precision(task='multiclass', num_classes=self.num_classes, average=None),
                'recall': Recall(task='multiclass', num_classes=self.num_classes, average=None),
            }  # test is done elsewhere manually
        }
        return

    def repeat_batch_encoding(self, batch_encoding: BatchEncoding, *repeats_per_dimension) -> BatchEncoding:
        """
        torch.repeat for each value in the batch_encoding
        """
        return BatchEncoding(
            {
                key: value.repeat(*repeats_per_dimension)
                for key, value in batch_encoding.items()
            }
        )

    def forward(
            self,
            batched_resume: Dict[str, dict],
            batched_job: Dict[str, dict],
            batched_resume_taxon_encoding: BatchEncoding,
            batched_job_taxon_encoding: BatchEncoding,
        ):
        """
        Basically, we are batching in the "dict keys" dimension
        e.g. consider a batch of size B
        batched_resume: {
            "desired_city": {
                "encoded_key_name": {},  # BatchEncoding of the string "desired_city". Repeated for B batches so Shape token_id=B * seq_length, token_type_ids=B * seq_length, etc.
                "encoded_values": {},  # BatchEncoding of the batched values for B resumes. Shape token_id = B * seq_length
            },
            ...
        }
        resume_taxon_encoding: BatchEncoding for the word "Resume" representing the document we are encoding. Shape token_id=1 * seq_length, etc.
        batched_resume_taxon_encoding: BatchEncoding by repeating the above. Shape token_id = B * seq_length
        """
        # taxon_embedding
        # batched_resume_taxon_tokens = self.repeat_batch_encoding(resume_taxon_encoding, (batch_size, 1))  # convert to B * seq_length
        geek_taxon = self.bert(
            **batched_resume_taxon_encoding.to(self.bert.device)
        ).last_hidden_state  # B * seq_length * word_embedding_size
        geek_taxon = self.geek_pool(geek_taxon).squeeze(1)  # B * word_embedding_size

        # batched_job_taxon_tokens = self.repeat_batch_encoding(job_taxon_encoding, (batch_size, 1))  # convert to B * seq_length
        job_taxon = self.bert(
            **batched_job_taxon_encoding.to(self.bert.device)
        ).last_hidden_state  # B * seq_length * word_embedding_size
        job_taxon = self.job_pool(job_taxon).squeeze(1)  # B * word_embedding_size
        
        # key_ewmbedding
        resume_keys = list(batched_resume.keys())
        geek_resume_key_embeddings = []
        for k in resume_keys:
            # key_encoding = batched_resume[k]["encoded_key_name"]
            # batched_key_encoding = self.repeat_batch_encoding(key_encoding, (batch_size, 1))  # convert to B * seq_length
            batched_key_encoding = batched_resume[k]["encoded_key_name"]
            geek_resume_key_embedding = self.bert(
                **batched_key_encoding.to(self.bert.device)
            ).last_hidden_state  # B * seq_length * word_embedding_size
            geek_resume_key_embedding = self.geek_pool(geek_resume_key_embedding).squeeze(1)  # B * word_embedding_size
            geek_resume_key_embeddings.append(geek_resume_key_embedding)

        job_keys = list(batched_job.keys())
        job_key_embeddings = []
        for k in job_keys:
            # key_encoding = batched_job[k]["encoded_key_name"]
            # batched_key_encoding = self.repeat_batch_encoding(key_encoding, (batch_size, 1))
            batched_key_encoding = batched_job[k]["encoded_key_name"]
            job_key_embedding = self.bert(
                **batched_key_encoding.to(self.bert.device)
            ).last_hidden_state
            job_key_embedding = self.job_pool(job_key_embedding).squeeze(1)
            job_key_embeddings.append(job_key_embedding)

        # value_ewmbedding
        geek_resume_value_embeddings = []
        for k in resume_keys:
            geek_resume_value_embedding = self.bert(
                **batched_resume[k]["encoded_values"].to(self.bert.device)
            ).last_hidden_state  # B * seq_length * word_embedding_size
            geek_resume_value_embedding = self.geek_pool(geek_resume_value_embedding).squeeze(1)  # B * word_embedding_size
            geek_resume_value_embeddings.append(geek_resume_value_embedding)

        job_value_embeddings = []
        for k in job_keys:
            job_value_embedding = self.bert(
                **batched_job[k]["encoded_values"].to(self.bert.device)
            ).last_hidden_state
            job_value_embedding = self.job_pool(job_value_embedding).squeeze(1)
            job_value_embeddings.append(job_value_embedding)

        # internal interaction interaction
        # let len(resume_keys) = 12, len(job_keys) = 11
        # InEXIT default: add
        _geek_key_embeddings = torch.stack(geek_resume_key_embeddings, dim=1)  # B * 12 * word_embedding_size
        _geek_value_embeddings = torch.stack(geek_resume_value_embeddings, dim=1)  # B * 12 * word_embedding_size
        geek = _geek_key_embeddings + _geek_value_embeddings  # B * 12 * word_embedding_size
        _job_key_embeddings = torch.stack(job_key_embeddings, dim=1)  # B * 11 * word_embedding_size
        _job_value_embeddings = torch.stack(job_value_embeddings, dim=1)  # B * 11 * word_embedding_size
        job = _job_key_embeddings + _job_value_embeddings  # B * 11 * word_embedding_size

        geek = self.internal_encoder(geek)
        job = self.internal_encoder(job)

        # InEXIT default: add the taxon token with the rest AGAIN
        num_geek_features = len(resume_keys)
        num_job_features = len(job_keys)
        geek = torch.repeat_interleave(geek_taxon.unsqueeze(1), repeats=num_geek_features, dim=1) + geek # geek_taxon=batch_size * word_embedding -> batch_size * 12 * word_embedding_size
        job = torch.repeat_interleave(job_taxon.unsqueeze(1), repeats=num_job_features, dim=1) + job # batch_size * 11 * word_embedding_size

        geek_job = torch.cat([geek, job], dim=1) # batch_size * (12 + 11) * (2 * word_embedding_size)

        # external interaction
        geek_job = self.external_encoder(geek_job)

        # final representation
        geek_vec, job_vec = torch.split(geek_job, (num_geek_features, num_job_features), dim=1)
        
        geek_vec = self.geek_pool(geek_vec).squeeze(1)  # back to batch_size * word_embedding_size
        job_vec = self.job_pool(job_vec).squeeze(1)

        # prediction layer
        x = torch.cat([job_vec, geek_vec, job_vec - geek_vec], dim=1)
        output = self.mlp(x).squeeze(1)

        return ModelOutput(
            output=output,
            batched_resume_representation=geek_vec,
            batched_job_representation=job_vec,
        )

    def training_step(self, batch, batch_idx) -> torch.Tensor:
        batched_label = batch.pop("batched_label")
        model_output = self(**batch)
        logit = model_output.output
        
        # assumes two classes
        if torch.sum(batched_label) == 0 or torch.sum(batched_label) == len(batched_label):
            # use BCE to be consistent with the original paper
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                input=logit,
                target=batched_label.float().to(logit.device),
            )
        else:
            # BCE requires a weight for each SAMPLE instead of class
            negative_weight = torch.tensor(1.0).to(batched_label.device)
            positive_weight = torch.sum(batched_label == 0) / torch.sum(batched_label)
            max_weight = torch.max(negative_weight, positive_weight)
            negative_weight /= max_weight
            positive_weight /= max_weight
            weight_for_each_sample = torch.where(
                batched_label == 0,
                negative_weight,
                positive_weight
            )
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                input=logit,
                target=batched_label.float().to(logit.device),
                weight=weight_for_each_sample.to(logit.device)
            )
        
        ### optimizer step.
        # Essential since scheduler from transformer does not directly work with pl lightning, we need to manually step
        optimizer = self.optimizers()
        loss /= self.args.gradient_accumulation_steps
        self.manual_backward(loss)
        # clip gradients
        self.clip_gradients(optimizer, gradient_clip_val=10.0, gradient_clip_algorithm="norm")
        # accumulate gradients of N batches
        if (batch_idx + 1) % self.args.gradient_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
        scheduler = self.lr_schedulers()
        scheduler.step()

        # logging to tensorboard by default. 
        # You will need to setup a folder `lightning_logs` in advance
        # also used to print to console
        self.log("train/loss", loss)
        with torch.no_grad():
            # class_preds = torch.argmax(pred, dim=1)
            class_preds = torch.sigmoid(logit) > 0.5
            self._log_metrics(class_preds, batched_label)
        return loss

    def validation_step(self, batch, batch_idx):
        # model.eval() and torch.no_grad() are called automatically for validation
        batched_label = batch.pop("batched_label")
        model_output = self(**batch)
        logits = model_output.output
        
        # assumes two classes
        if torch.sum(batched_label) == 0 or torch.sum(batched_label) == len(batched_label):
            ## all 0 or all 1
            val_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                input=logits,
                target=batched_label.float().to(logits.device),
            )
        else:
            # BCE requires a weight for each SAMPLE instead of class
            negative_weight = torch.tensor(1.0).to(batched_label.device)
            positive_weight = torch.sum(batched_label == 0) / torch.sum(batched_label)
            max_weight = torch.max(negative_weight, positive_weight)
            negative_weight /= max_weight
            positive_weight /= max_weight
            weight_for_each_sample = torch.where(
                batched_label == 0,
                negative_weight,
                positive_weight
            )

            val_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                input=logits,
                target=batched_label.float().to(logits.device),
                weight=weight_for_each_sample.to(logits.device)
            )

        self.log('val/loss', val_loss, sync_dist=True)
        # class_preds = torch.argmax(pred, dim=1)
        class_preds = torch.sigmoid(logits) > 0.5
        self._log_metrics(class_preds, batched_label, mode='val')
        return val_loss
    
    def predict_step(self, batch, batch_idx):
        # usually batch would only contain x, as you have no y
        # here for simplicity I justed used test_loader for input, hence we also have y
        if "batched_label" in batch:
            _ = batch.pop("batched_label")
        
        model_output = self(**batch)
        logits = model_output.output
        # class_preds = torch.argmax(pred, dim=1)
        class_preds = torch.sigmoid(logits) > 0.5
        return ModelOutput(
            logits=logits,
            class_preds=class_preds,
        )

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.args.lr,
            weight_decay=self.args.weight_decay,
        )
        # learning rate schedulers do not work due to pytorch version
        # e.g. The provided lr scheduler `OneCycleLR` doesn't follow PyTorch's LRScheduler API.
        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(self.args.num_warmup_steps * self.trainer.max_steps),
            num_training_steps=self.trainer.max_steps
        )
        return [optimizer], [scheduler]