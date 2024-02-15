from torchmetrics import Precision, Recall, F1Score
from transformers import AutoModel
from transformers.utils import ModelOutput
from transformers import get_linear_schedule_with_warmup
from typing import Dict
from dataclasses import dataclass, field
from src.model.base import BaseModel
import torch


@dataclass
class DPGNNModelArguments:
    lr: float = field(default=1e-5)
    gradient_accumulation_steps: int = field(default=2)
    weight_decay: float = field(default=1e-4)
    num_warmup_steps: float = field(default=0.05)
    num_resume_features: int = field(
        default=12, metadata={"help": "Number of keys in resume"}
    )
    num_job_features: int = field(
        default=11, metadata={"help": "Number of keys in job description"}
    )
    word_emb_dim: int = field(default=768)
    hidden_size: int = field(default=768)
    num_heads: int = field(default=8)
    num_encoder_layers: int = field(
        default=1, metadata={"help": "Number of transformer layers for self-attention"}
    )
    dropout: float = field(default=0.1)
    temperature: float = field(
        default=0.2,  # according to the original DPGNN repo
        metadata={"help": "Temperature for logit"},
    )
    pretrained_encoder: str = field(default="bert-base-chinese")
    mul_weight: float = field(
        default=0.05,  # according to the original DPGNN repo
        metadata={"help": "Weight for mutual loss"},
    )
    reg_weight: float = field(
        default=0.1,
        metadata={"help": "Weight for regularization loss"},
    )
    bpr_loss_version: str = field(
        default="original",  # original = copied from DPGNN repo, modified = see comment in our implementation
        metadata={"help": "BPR loss function"},
    )
    remove_passive_embedding: bool = field(
        default=False,  # according to the original DPGNN repo
        metadata={"help": "Whether to remove passive embedding"},
    )
    do_normalize: bool = field(
        default=True,
        metadata={"help": "Whether to normalize the embedding"},
    )
    embedding_method: str = field(
        default="cls_token",  # according to the original DPGNN repo
        metadata={"help": "Method to compute embedding from bert output"},
    )

    def __post_init__(self):
        assert self.embedding_method in [
            "average_pool",
            "cls_token",
        ], f"embedding_method {self.embedding_method} not supported"
        assert self.bpr_loss_version in [
            "original",
            "modified",
        ], f"bpr_loss_version {self.bpr_loss_version} not supported"
        return


class BPRLoss(torch.nn.Module):
    def __init__(self, gamma=1e-10):
        super(BPRLoss, self).__init__()
        self.gamma = gamma

    def forward(self, pos_score, neg_score):
        loss = -torch.log(self.gamma + torch.sigmoid(pos_score - neg_score)).mean()
        return loss


class EmbLoss(torch.nn.Module):
    """EmbLoss, regularization on embeddings"""

    def __init__(self, norm=2):
        super(EmbLoss, self).__init__()
        self.norm = norm

    def forward(self, *embeddings):
        emb_loss = torch.zeros(1).to(embeddings[-1].device)
        for embedding in embeddings:
            emb_loss += torch.norm(embedding, p=self.norm)
        emb_loss /= embeddings[-1].shape[0]
        return emb_loss


@dataclass
class ActivePassiveEmbeddingOutput:
    """wrapper for outputing two embedding vectors per resume/job"""

    active_representation: torch.Tensor
    passive_representation: torch.Tensor


class DPGNNModel(BaseModel):
    """dpgnn modified to work with unseen resume/jobs"""

    def __init__(self, args: DPGNNModelArguments):
        super(DPGNNModel, self).__init__()
        self.automatic_optimization = (
            False  # we manually do optimizer.step() and scheduler.step()
        )

        self.args = args
        self.num_classes = 2
        self.bpr_loss_fn = BPRLoss()
        self.reg_loss_fn = EmbLoss()
        self.mutual_loss = torch.nn.CrossEntropyLoss()

        self.embedding_method = args.embedding_method

        self.bert = AutoModel.from_pretrained(args.pretrained_encoder)

        for param in self.bert.parameters():
            param.requires_grad = True

        # resume
        self.resume_active_encoder = torch.nn.TransformerEncoder(
            torch.nn.TransformerEncoderLayer(
                d_model=args.word_emb_dim,
                nhead=args.num_heads,
                dim_feedforward=args.hidden_size,
                dropout=args.dropout,
                batch_first=True,
                norm_first=False,
            ),
            num_layers=args.num_encoder_layers,
        )
        self.resume_active_merge_layer = torch.nn.Linear(
            self.args.num_resume_features, 1
        )
        self.resume_passive_encoder = torch.nn.TransformerEncoder(
            torch.nn.TransformerEncoderLayer(
                d_model=args.word_emb_dim,
                nhead=args.num_heads,
                dim_feedforward=args.hidden_size,
                dropout=args.dropout,
                batch_first=True,
                norm_first=False,
            ),
            num_layers=args.num_encoder_layers,
        )
        self.resume_passive_merge_layer = torch.nn.Linear(
            self.args.num_resume_features, 1
        )

        self.job_active_encoder = torch.nn.TransformerEncoder(
            torch.nn.TransformerEncoderLayer(
                d_model=args.word_emb_dim,
                nhead=args.num_heads,
                dim_feedforward=args.hidden_size,
                dropout=args.dropout,
                batch_first=True,
                norm_first=False,
            ),
            num_layers=args.num_encoder_layers,
        )
        self.job_active_merge_layer = torch.nn.Linear(self.args.num_job_features, 1)
        self.job_passive_encoder = torch.nn.TransformerEncoder(
            torch.nn.TransformerEncoderLayer(
                d_model=args.word_emb_dim,
                nhead=args.num_heads,
                dim_feedforward=args.hidden_size,
                dropout=args.dropout,
                batch_first=True,
                norm_first=False,
            ),
            num_layers=args.num_encoder_layers,
        )
        self.job_passive_merge_layer = torch.nn.Linear(self.args.num_job_features, 1)

        self.metrics = {
            "train": {
                "f1": F1Score(
                    task="multiclass", num_classes=self.num_classes, average="weighted"
                ),
                "precision": Precision(
                    task="multiclass", num_classes=self.num_classes, average=None
                ),
                "recall": Recall(
                    task="multiclass", num_classes=self.num_classes, average=None
                ),
            },
            "val": {
                "f1": F1Score(
                    task="multiclass", num_classes=self.num_classes, average="weighted"
                ),
                "precision": Precision(
                    task="multiclass", num_classes=self.num_classes, average=None
                ),
                "recall": Recall(
                    task="multiclass", num_classes=self.num_classes, average=None
                ),
            },  # test is done elsewhere manually
        }
        return

    def average_pool(
        self, last_hidden_states: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        last_hidden = last_hidden_states.masked_fill(
            ~attention_mask[..., None].bool(), 0.0
        )
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

    def _get_encoder_embedding(self, model, batched_encoding: dict) -> torch.Tensor:
        """given a pretrained encoder model (e.g. bert), how to get the embedding vector"""
        embedding_vec = model(
            **batched_encoding.to(model.device)
        ).last_hidden_state  # B * seq_length * word_embedding_size

        if self.embedding_method == "cls_token":
            embedding_vec = embedding_vec[:, 0, :]  # B * word_embedding_size
        elif self.embedding_method == "average_pool":
            embedding_vec = self.average_pool(
                embedding_vec, attention_mask=batched_encoding["attention_mask"]
            )  # B * word_embedding_size
        else:
            raise NotImplementedError
        return embedding_vec

    def forward(
        self,
        batched_dict_data: Dict[str, dict],
        data_type: str,
    ):
        """embed the batched_dict_data. data_type is either "resume" or "job"
        e.g. consider a batch of size B for data_type "resume"
        batched_dict_data: {
            "desired_city": {
                "encoded_key_values": {},  # BatchEncoding of the batched taxon_token + key + values for B resumes. Shape token_id = B * seq_length
            },
            ...
        }
        """
        assert data_type in ["resume", "job"]

        dict_keys = list(batched_dict_data.keys())
        data_kv_embeddings = []
        for k in dict_keys:
            batched_kv_encoding = batched_dict_data[k]["encoded_key_values"]
            # B * word_embedding_size
            data_kv_embedding = self._get_encoder_embedding(
                self.bert, batched_kv_encoding
            )
            data_kv_embeddings.append(data_kv_embedding)

        ### compute interaction beween each k,v pair in a resume or job
        ### then merge to produce a single vector representation
        # let len(dict_keys) = 12, len(job_keys) = 11
        active_layer = (
            self.resume_active_encoder
            if data_type == "resume"
            else self.job_active_encoder
        )
        active_merge_layer = (
            self.resume_active_merge_layer
            if data_type == "resume"
            else self.job_active_merge_layer
        )
        passive_layer = (
            self.resume_passive_encoder
            if data_type == "resume"
            else self.job_passive_encoder
        )
        passive_merge_layer = (
            self.resume_passive_merge_layer
            if data_type == "resume"
            else self.job_passive_merge_layer
        )

        # model active embedding
        _data_kv_embeddings = torch.stack(
            data_kv_embeddings, dim=1
        )  # B * 12 * word_embedding_size
        _active_hidden_states = active_layer(
            _data_kv_embeddings
        )  # B * 12 * word_embedding_size
        _active_hidden_states = torch.permute(
            _active_hidden_states, (0, 2, 1)
        )  # B * word_embedding_size * 12
        active_vec = active_merge_layer(_active_hidden_states).squeeze(
            -1
        )  # B * word_embedding_size

        # model passive embedding
        _passive_hidden_states = passive_layer(
            _data_kv_embeddings
        )  # B * 12 * word_embedding_size
        _passive_hidden_states = torch.permute(
            _passive_hidden_states, (0, 2, 1)
        )  # B * word_embedding_size * 12
        passive_vec = passive_merge_layer(_passive_hidden_states).squeeze(
            -1
        )  # B * word_embedding_size

        if self.args.do_normalize:
            active_vec = torch.nn.functional.normalize(active_vec, p=2, dim=1)
            passive_vec = torch.nn.functional.normalize(passive_vec, p=2, dim=1)

        return ActivePassiveEmbeddingOutput(
            active_representation=active_vec,
            passive_representation=passive_vec,
        )

    def _get_info_nce(
        self,
        active_vec: torch.Tensor,
        passive_vec: torch.Tensor,
    ):
        similarity_matrix = torch.matmul(active_vec, passive_vec.T)  # B * B
        bsz = similarity_matrix.shape[0]

        # diagonal elements are positive pairs, all the rest are negative pairs
        mask = torch.eye(bsz, dtype=torch.bool).to(similarity_matrix.device)
        positives = similarity_matrix[mask].view(bsz, -1)
        negatives = similarity_matrix[~mask].view(bsz, -1)

        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(logits.device)
        logits = logits / self.args.temperature
        return logits, labels

    def training_step(self, batch, batch_idx) -> torch.Tensor:
        """batch still consist of batched resume/job in the key dimension, BUT we have two differences:
        1. added hard negative (if non-exists, randomly sample) for BOTH resume and jobs
        2. the i-th positive resume is matched with the i-th positive job
        3. the i-th neg resume is (un)matched with the i-th positive job, and vice versa
        """
        batched_resume = batch["batched_resume"]
        batched_job = batch["batched_job"]
        batched_resume_negatives = batch.get("batched_resume_negatives", {})
        batched_job_negatives = batch.get("batched_job_negatives", {})

        batched_resume_vec = self.forward(batched_resume, data_type="resume")
        batched_job_vec = self.forward(batched_job, data_type="job")
        batched_resume_negatives_vec = self.forward(
            batched_resume_negatives, data_type="resume"
        )
        batched_job_negatives_vec = self.forward(batched_job_negatives, data_type="job")

        if self.args.remove_passive_embedding:
            resume_active_vec = batched_resume_vec.active_representation
            resume_passive_vec = resume_active_vec
            job_active_vec = batched_job_vec.active_representation
            job_passive_vec = job_active_vec
            resume_negatives_active_vec = (
                batched_resume_negatives_vec.active_representation
            )
            resume_negatives_passive_vec = resume_negatives_active_vec
            job_negatives_active_vec = batched_job_negatives_vec.active_representation
            job_negatives_passive_vec = job_negatives_active_vec
        else:
            resume_active_vec = batched_resume_vec.active_representation
            resume_passive_vec = batched_resume_vec.passive_representation
            job_active_vec = batched_job_vec.active_representation
            job_passive_vec = batched_job_vec.passive_representation
            resume_negatives_active_vec = (
                batched_resume_negatives_vec.active_representation
            )
            resume_negatives_passive_vec = (
                batched_resume_negatives_vec.passive_representation
            )
            job_negatives_active_vec = batched_job_negatives_vec.active_representation
            job_negatives_passive_vec = batched_job_negatives_vec.passive_representation

        r_pos = torch.mul(resume_active_vec, job_passive_vec).sum(dim=1)
        s_pos = torch.mul(resume_passive_vec, job_active_vec).sum(dim=1)

        r_neg1 = torch.mul(resume_active_vec, job_negatives_passive_vec).sum(dim=1)
        s_neg1 = torch.mul(resume_passive_vec, job_negatives_active_vec).sum(dim=1)

        r_neg2 = torch.mul(resume_negatives_active_vec, job_passive_vec).sum(dim=1)
        s_neg2 = torch.mul(resume_negatives_passive_vec, job_active_vec).sum(dim=1)

        if self.args.bpr_loss_version == "original":
            ### original implementation problem: you can have (r_pos + s_pos) = 10, and get away with (r_neg1 + s_neg1) = 11 and (r_neg2 + s_neg2) = 8
            bpr_loss = self.bpr_loss_fn(
                2 * r_pos + 2 * s_pos, r_neg1 + s_neg1 + r_neg2 + s_neg2
            )
        else:
            bpr_loss_1 = self.bpr_loss_fn(r_pos + s_pos, r_neg1 + s_neg1)
            bpr_loss_2 = self.bpr_loss_fn(r_pos + s_pos, r_neg2 + s_neg2)
            bpr_loss = (bpr_loss_1 + bpr_loss_2) / 2

        ### removed graph embedding loss as it is requires knowing ALL resume and jobs in advance
        ### this will be "cheating" in our case as we are holding out resume and jobs
        reg_loss = torch.zeros(1).to(bpr_loss.device)        

        if self.args.remove_passive_embedding:
            mul_user_loss = torch.zeros(1).to(bpr_loss.device)
            mul_job_loss = torch.zeros(1).to(bpr_loss.device)
        else:
            # INFO NCE loss
            logits_user, labels = self._get_info_nce(
                resume_active_vec,
                resume_passive_vec,
            )
            mul_user_loss = self.mutual_loss(logits_user, labels)

            logits_job, labels = self._get_info_nce(
                job_active_vec,
                job_passive_vec,
            )
            mul_job_loss = self.mutual_loss(logits_job, labels)

        loss = (
            bpr_loss + self.args.mul_weight * (mul_user_loss + mul_job_loss) + self.args.reg_weight * reg_loss
        )

        ### optimizer step.
        # Essential since scheduler from transformer does not directly work with pl lightning, we need to manually step
        optimizer = self.optimizers()
        loss /= self.args.gradient_accumulation_steps
        self.manual_backward(loss)
        # clip gradients
        self.clip_gradients(
            optimizer, gradient_clip_val=10.0, gradient_clip_algorithm="norm"
        )
        # accumulate gradients of N batches
        if (batch_idx + 1) % self.args.gradient_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
        scheduler = self.lr_schedulers()
        scheduler.step()

        # logging to tensorboard by default.
        # You will need to setup a folder `lightning_logs` in advance
        # also used to print to console
        with torch.no_grad():
            self.log("train/loss", loss)
            self.log("train/bpr_loss", bpr_loss)
            self.log("train/reg_loss", reg_loss)
            self.log("train/mul_user_loss", mul_user_loss)
            self.log("train/mul_job_loss", mul_job_loss)

            geek_vec_length = torch.norm(resume_active_vec, dim=1)
            geek_vec_avg_length = torch.mean(geek_vec_length)
            job_vec_length = torch.norm(job_active_vec, dim=1)
            job_vec_avg_length = torch.mean(job_vec_length)
            self.log("train/geek_vec_avg_length", geek_vec_avg_length)
            self.log("train/job_vec_avg_length", job_vec_avg_length)

            num_positive = resume_active_vec.shape[0]
            positive_score = 0.5 * (r_pos + s_pos)
            normalized_positive_score = torch.sigmoid(positive_score)
            positive_class_preds = normalized_positive_score > 0.5

            # negative samples
            negative_score_1 = 0.5 * (r_neg1 + s_neg1)
            normalized_negative_score_1 = torch.sigmoid(negative_score_1)
            negative_class_preds_1 = normalized_negative_score_1 > 0.5
            negative_score_2 = 0.5 * (r_neg2 + s_neg2)
            normalized_negative_score_2 = torch.sigmoid(negative_score_2)
            negative_class_preds_2 = normalized_negative_score_2 > 0.5
            negative_class_preds = torch.cat(
                [negative_class_preds_1, negative_class_preds_2], dim=0
            )

            self.log("train/max_pos_score", torch.max(positive_score))
            self.log("train/min_pos_score", torch.min(positive_score))
            self.log("train/mean_pos_score", torch.mean(positive_score))
            self.log("train/example_pos_score", positive_score[0])
            self.log("train/max_neg_score_1", torch.max(negative_score_1))
            self.log("train/min_neg_score_1", torch.min(negative_score_1))
            self.log("train/mean_neg_score_1", torch.mean(negative_score_1))
            self.log("train/example_neg_score_1", negative_score_1[0])
            self.log("train/max_neg_score_2", torch.max(negative_score_2))
            self.log("train/min_neg_score_2", torch.min(negative_score_2))
            self.log("train/mean_neg_score_2", torch.mean(negative_score_2))
            self.log("train/example_neg_score_2", negative_score_2[0])

            all_labels = torch.cat(
                [torch.ones(num_positive), torch.zeros(negative_class_preds.shape[0])],
                dim=0,
            )
            all_preds = torch.cat([positive_class_preds, negative_class_preds], dim=0)
            self._log_metrics(all_preds, all_labels)
        return loss

    def validation_step(self, batch, batch_idx):
        # model.eval() and torch.no_grad() are called automatically for validation
        batched_resume = batch["batched_resume"]
        batched_job = batch["batched_job"]
        batched_resume_negatives = batch.get("batched_resume_negatives", {})
        batched_job_negatives = batch.get("batched_job_negatives", {})

        batched_resume_vec = self.forward(batched_resume, data_type="resume")
        batched_job_vec = self.forward(batched_job, data_type="job")
        batched_resume_negatives_vec = self.forward(
            batched_resume_negatives, data_type="resume"
        )
        batched_job_negatives_vec = self.forward(batched_job_negatives, data_type="job")

        resume_active_vec = batched_resume_vec.active_representation
        resume_passive_vec = batched_resume_vec.passive_representation
        job_active_vec = batched_job_vec.active_representation
        job_passive_vec = batched_job_vec.passive_representation
        resume_negatives_active_vec = batched_resume_negatives_vec.active_representation
        resume_negatives_passive_vec = (
            batched_resume_negatives_vec.passive_representation
        )
        job_negatives_active_vec = batched_job_negatives_vec.active_representation
        job_negatives_passive_vec = batched_job_negatives_vec.passive_representation

        r_pos = torch.mul(resume_active_vec, job_passive_vec).sum(dim=1)
        s_pos = torch.mul(resume_passive_vec, job_active_vec).sum(dim=1)

        r_neg1 = torch.mul(resume_active_vec, job_negatives_passive_vec).sum(dim=1)
        s_neg1 = torch.mul(resume_passive_vec, job_negatives_active_vec).sum(dim=1)

        r_neg2 = torch.mul(resume_negatives_active_vec, job_passive_vec).sum(dim=1)
        s_neg2 = torch.mul(resume_negatives_passive_vec, job_active_vec).sum(dim=1)

        if self.args.bpr_loss_version == "original":
            ### original implementation problem: you can have (r_pos + s_pos) = -135, and get away with (r_neg1 + s_neg1) = -141 and (r_neg2 + s_neg2) = -130
            bpr_loss = self.bpr_loss_fn(
                2 * r_pos + 2 * s_pos, r_neg1 + s_neg1 + r_neg2 + s_neg2
            )
        else:
            bpr_loss_1 = self.bpr_loss_fn(2 * r_pos + 2 * s_pos, r_neg1 + s_neg1)
            bpr_loss_2 = self.bpr_loss_fn(2 * r_pos + 2 * s_pos, r_neg2 + s_neg2)
            bpr_loss = (bpr_loss_1 + bpr_loss_2) / 2

        ### removed embedding loss as it is requires knowing ALL resume and jobs in advance
        ### this will be "cheating" in our case as we are holding out resume and jobs
        reg_loss = torch.zeros(1).to(bpr_loss.device)

        # INFO NCE loss
        logits_user, labels = self._get_info_nce(
            resume_active_vec,
            resume_passive_vec,
        )
        mul_user_loss = self.mutual_loss(logits_user, labels)

        logits_job, labels = self._get_info_nce(
            job_active_vec,
            job_passive_vec,
        )
        mul_job_loss = self.mutual_loss(logits_job, labels)

        val_loss = (
            bpr_loss + self.args.mul_weight * (mul_user_loss + mul_job_loss) + self.args.reg_weight *  reg_loss
        )

        ### logging
        self.log("val/loss", val_loss, sync_dist=True)

        num_positive = resume_active_vec.shape[0]
        positive_score = 0.5 * (r_pos + s_pos)
        normalized_positive_score = torch.sigmoid(positive_score)
        positive_class_preds = normalized_positive_score > 0.5

        # negative samples
        negative_score_1 = 0.5 * (r_neg1 + s_neg1)
        normalized_negative_score_1 = torch.sigmoid(negative_score_1)
        negative_class_preds_1 = normalized_negative_score_1 > 0.5
        negative_score_2 = 0.5 * (r_neg2 + s_neg2)
        normalized_negative_score_2 = torch.sigmoid(negative_score_2)
        negative_class_preds_2 = normalized_negative_score_2 > 0.5
        negative_class_preds = torch.cat(
            [negative_class_preds_1, negative_class_preds_2], dim=0
        )

        all_labels = torch.cat(
            [torch.ones(num_positive), torch.zeros(negative_class_preds.shape[0])],
            dim=0,
        )
        all_preds = torch.cat([positive_class_preds, negative_class_preds], dim=0)
        self._log_metrics(all_preds, all_labels, mode="val")
        return val_loss

    def predict_step(self, batch, batch_idx):
        # usually batch would only contain x, as you have no y
        # here for simplicity I justed used test_loader for input, hence we also have y
        if "batched_label" in batch:
            _ = batch.pop("batched_label")

        batched_resume = batch["batched_resume"]
        batched_job = batch["batched_job"]

        batched_resume_vec = self.forward(batched_resume, data_type="resume")
        batched_job_vec = self.forward(batched_job, data_type="job")

        resume_active_vec = batched_resume_vec.active_representation
        resume_passive_vec = batched_resume_vec.passive_representation
        job_active_vec = batched_job_vec.active_representation
        job_passive_vec = batched_job_vec.passive_representation

        positive_score = torch.sum(
            resume_active_vec * job_passive_vec, axis=1
        ) + torch.sum(resume_passive_vec * job_active_vec, axis=1)
        positive_score /= 2
        normalized_positive_score = torch.sigmoid(positive_score)
        class_preds = normalized_positive_score > 0.5

        return ModelOutput(
            logits=positive_score,
            class_preds=class_preds,
            batched_resume_representation=resume_passive_vec,
            batched_job_representation=job_passive_vec,
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
            num_training_steps=self.trainer.max_steps,
        )
        return [optimizer], [scheduler]
