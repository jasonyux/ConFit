from torchmetrics import Precision, Recall, F1Score
from transformers import AutoModel
from transformers.utils import ModelOutput
from transformers import get_linear_schedule_with_warmup
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from deepspeed.ops.adam import DeepSpeedCPUAdam
from src.model.base import BaseModel
import torch


@dataclass
class ConFitModelArguments:
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
    num_suit_encoder_layers: int = field(
        default=1,
        metadata={
            "help": "Number of transformer layers for self-attention for the suitability layer"
        },
    )
    do_both_rj_hard_neg: bool = field(
        default=False,
        metadata={
            "help": "Whether to do both resume-job and job-resume hard negative mining"
        },
    )
    # TODO: probably bad idea because for RANKING we use un-normalized score!
    do_normalize: bool = field(
        default=False, metadata={"help": "Whether to normalize ALL embeddings for loss calculations"}
    )
    coarse_training_portion: float = field(
        default=0.7,
        metadata={
            "help": "Portion of training steps to ONLY train coarse-grained loss. After which we train both coarse and fine-grained loss."
        },
    )
    finegrained_loss: str = field(
        default="noop",  # see __post_init__ for supported loss
        metadata={"help": "Which fine-grained loss to use"},
    )
    alpha: float = field(
        default=0.1,
        metadata={"help": "Weight for the fine-grained loss"},
    )
    beta: float = field(
        default=1.0,
        metadata={
            "help": "Weight for the fine-grained MSE (yes, not KL) divergence loss used by BCEwconstraint"
        },
    )
    dropout: float = field(default=0.1)
    temperature: float = field(default=1.0, metadata={"help": "Temperature for logit"})
    pretrained_encoder: str = field(default="bert-base-chinese")
    share_encoder: bool = field(
        default=True,
        metadata={"help": "Whether to share the same encoder for resume and job"},
    )
    embedding_method: str = field(
        default="average_pool",
        metadata={"help": "Method to compute embedding from bert output"},
    )
    cpu_adam: bool = field(
        default=False,
        metadata={"help": "Whether to use cpu for adam optimizer"},
    )

    def __post_init__(self):
        assert self.embedding_method in [
            "average_pool",
            "cls_token",
        ], f"embedding_method {self.embedding_method} not supported"
        assert self.finegrained_loss in [
            "noop",
        ], f"finegrained_loss {self.finegrained_loss} not supported"
        # sutble but potentially important changes
        if self.do_normalize:
            print("Normalizing ALL embeddings for loss calculations")
        return


@dataclass
class DoubleEmbeddingOutput:
    """wrapper for outputing two embedding vectors per resume/job"""

    coarse_embedding: torch.Tensor
    finegrained_embedding: torch.Tensor


class ConFitModel(BaseModel):
    """rag like model but with supervised loss"""

    def __init__(self, args: ConFitModelArguments):
        super(ConFitModel, self).__init__()
        self.automatic_optimization = (
            False  # we manually do optimizer.step() and scheduler.step()
        )

        self.args = args
        self.num_classes = 2
        self.embedding_method = args.embedding_method

        if args.share_encoder:
            self.bert_resume = AutoModel.from_pretrained(args.pretrained_encoder)
            self.bert_job = self.bert_resume
        else:
            self.bert_resume = AutoModel.from_pretrained(args.pretrained_encoder)
            self.bert_job = AutoModel.from_pretrained(args.pretrained_encoder)

        for param in self.bert_resume.parameters():
            param.requires_grad = True
        for param in self.bert_job.parameters():
            param.requires_grad = True

        self.resume_interaction_encoder = torch.nn.TransformerEncoder(
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
        self.resume_merge_layer = torch.nn.Linear(self.args.num_resume_features, 1)
        self.resume_suitablity_encoder = torch.nn.TransformerEncoder(
            torch.nn.TransformerEncoderLayer(
                d_model=args.word_emb_dim,
                nhead=args.num_heads,
                dim_feedforward=args.hidden_size,
                dropout=args.dropout,
                batch_first=True,
                norm_first=False,
            ),
            num_layers=1,
        )
        self.resume_suitability_merge_layer = torch.nn.Linear(
            self.args.num_resume_features, 1
        )

        self.job_interaction_encoder = torch.nn.TransformerEncoder(
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
        self.job_merge_layer = torch.nn.Linear(self.args.num_job_features, 1)
        self.job_suitablity_encoder = torch.nn.TransformerEncoder(
            torch.nn.TransformerEncoderLayer(
                d_model=args.word_emb_dim,
                nhead=args.num_heads,
                dim_feedforward=args.hidden_size,
                dropout=args.dropout,
                batch_first=True,
                norm_first=False,
            ),
            num_layers=1,
        )
        self.job_suitability_merge_layer = torch.nn.Linear(
            self.args.num_job_features, 1
        )

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
        embedding_model = self.bert_resume if data_type == "resume" else self.bert_job

        dict_keys = list(batched_dict_data.keys())
        data_kv_embeddings = []
        for k in dict_keys:
            batched_kv_encoding = batched_dict_data[k]["encoded_key_values"]
            # B * word_embedding_size
            data_kv_embedding = self._get_encoder_embedding(
                embedding_model, batched_kv_encoding
            )
            data_kv_embeddings.append(data_kv_embedding)

        ### compute interaction beween each k,v pair in a resume or job
        ### then merge to produce a single vector representation
        # let len(dict_keys) = 12, len(job_keys) = 11
        interaction_layer = (
            self.resume_interaction_encoder
            if data_type == "resume"
            else self.job_interaction_encoder
        )
        merge_layer = (
            self.resume_merge_layer if data_type == "resume" else self.job_merge_layer
        )
        finegrained_layer = (
            self.resume_suitablity_encoder
            if data_type == "resume"
            else self.job_suitablity_encoder
        )
        finegrained_merge_layer = (
            self.resume_suitability_merge_layer
            if data_type == "resume"
            else self.job_suitability_merge_layer
        )

        # model coarse-grained embedding
        _data_kv_embeddings = torch.stack(
            data_kv_embeddings, dim=1
        )  # B * 12 * word_embedding_size
        _data_kv_embeddings = interaction_layer(
            _data_kv_embeddings
        )  # B * 12 * word_embedding_size
        _coarse_grained_embeddings = torch.permute(
            _data_kv_embeddings, (0, 2, 1)
        )  # B * word_embedding_size * 12
        coarse_vec = merge_layer(_coarse_grained_embeddings).squeeze(
            -1
        )  # B * word_embedding_size

        # model fine-grained embedding
        _finegrained_embeddings = finegrained_layer(
            _data_kv_embeddings
        )  # B * 12 * word_embedding_size
        _finegrained_embeddings = torch.permute(
            _finegrained_embeddings, (0, 2, 1)
        )  # B * word_embedding_size * 12
        finegrained_vec = finegrained_merge_layer(_finegrained_embeddings).squeeze(
            -1
        )  # B * word_embedding_size

        if self.args.num_suit_encoder_layers == 0:
            finegrained_vec = coarse_vec

        if self.args.do_normalize:
            coarse_vec = torch.nn.functional.normalize(coarse_vec, p=2, dim=1)
            finegrained_vec = torch.nn.functional.normalize(finegrained_vec, p=2, dim=1)

        return DoubleEmbeddingOutput(
            coarse_embedding=coarse_vec, finegrained_embedding=finegrained_vec
        )

    def compute_finegrained_loss(
        self,
        batched_resume_hard_negatives_indices: List[int],
        batched_job_hard_negatives_indices: List[int],
        batched_resume_coarse_vec: torch.Tensor,
        batched_resume_finegrained_vec: torch.Tensor,
        batched_job_coarse_vec: torch.Tensor,
        batched_job_finegrained_vec: torch.Tensor,
        batched_job_hard_negatives_coarse_vec: Optional[torch.Tensor] = None,
        batched_job_hard_negatives_finegrained_vec: Optional[torch.Tensor] = None,
        batched_all_coarse_job: Optional[torch.Tensor] = None,
        batched_resume_hard_negatives_coarse_vec: Optional[torch.Tensor] = None,
        batched_resume_hard_negatives_finegrained_vec: Optional[torch.Tensor] = None,
        batched_all_coarse_resume: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """entry point for computing different fine-grained loss specified in self.args.finegrained_loss

        Args:
            batched_resume_hard_negatives_indices (List[int]): which job these resume_hard_neg corresponds to
            batched_job_hard_negatives_indices (List[int]): which resume these job_hard_neg corresponds to
            batched_resume_coarse_vec (torch.Tensor): _description_
            batched_resume_finegrained_vec (torch.Tensor): _description_
            batched_job_coarse_vec (torch.Tensor): _description_
            batched_job_finegrained_vec (torch.Tensor): _description_
            batched_job_hard_negatives_coarse_vec (Optional[torch.Tensor], optional): _description_. Defaults to None.
            batched_job_hard_negatives_finegrained_vec (Optional[torch.Tensor], optional): _description_. Defaults to None.
            batched_all_coarse_job (Optional[torch.Tensor], optional): Concatenation of batched_job_coarse_vec and batched_job_hard_negatives_coarse_vec. \
                This is mainly to save memory as we called concat before. Defaults to None.
            batched_resume_hard_negatives_coarse_vec (Optional[torch.Tensor], optional): _description_. Defaults to None.
            batched_resume_hard_negatives_finegrained_vec (Optional[torch.Tensor], optional): _description_. Defaults to None.
            batched_all_coarse_resume (Optional[torch.Tensor], optional): _description_. Defaults to None.

        Raises:
            NotImplementedError: _description_

        Returns:
            torch.Tensor: _description_
        """
        loss_fn = None
        if self.args.finegrained_loss == "noop":
            return torch.zeros(
                1, dtype=torch.float, device=batched_resume_coarse_vec.device
            )
        else:
            raise NotImplementedError(
                f"finegrained_loss {self.args.finegrained_loss} not implemented"
            )
        return loss_fn(
            batched_resume_hard_negatives_indices,
            batched_job_hard_negatives_indices,
            batched_resume_coarse_vec,
            batched_resume_finegrained_vec,
            batched_job_coarse_vec,
            batched_job_finegrained_vec,
            batched_job_hard_negatives_coarse_vec,
            batched_job_hard_negatives_finegrained_vec,
            batched_all_coarse_job,
            batched_resume_hard_negatives_coarse_vec,
            batched_resume_hard_negatives_finegrained_vec,
            batched_all_coarse_resume
        )

    def compute_coarsegrained_loss(
        self,
        batched_resume_coarse_vec: torch.Tensor,
        batched_job_coarse_vec: torch.Tensor,
        batched_resume_hard_negatives: dict,
        batched_resume_hard_negatives_indices: List[int],
        batched_job_hard_negatives: dict,
        batched_job_hard_negatives_indices: List[int],
        batched_resume_positives: dict,
        batched_resume_positive_indices: List[int],
        batched_job_positives: dict,
        batched_job_positive_indices: List[int],
    ):
        """where we compute contrastive objective

        Args:
            batched_resume_coarse_vec (torch.Tensor): _description_
            batched_job_coarse_vec (torch.Tensor): _description_
            batched_resume_hard_negatives (dict): _description_
            batched_resume_hard_negatives_indices (List[int]): _description_
            batched_job_hard_negatives (dict): _description_
            batched_job_hard_negatives_indices (List[int]): _description_
            batched_resume_positives (dict): _description_
            batched_resume_positive_indices (List[int]): _description_
            batched_job_positives (dict): _description_
            batched_job_positive_indices (List[int]): _description_

        Returns:
            _type_: _description_
        """
        batched_resume_hard_negatives_coarse_vec = None
        batched_resume_hard_negatives_finegrained_vec = None
        batched_all_coarse_resume = None
        batched_job_hard_negatives_coarse_vec = None
        batched_job_hard_negatives_finegrained_vec = None
        batched_all_coarse_job = None

        num_resume_hard_negs_used = 0
        num_job_hard_negs_used = 0
        num_resume_positive_used = 0
        num_job_positive_used = 0
        contrastive_loss = torch.zeros(
            1, dtype=torch.float, device=batched_resume_coarse_vec.device
        )
        if (
            len(batched_resume_hard_negatives) == 0
            and len(batched_job_hard_negatives) == 0
        ):
            # may happen if no pairing satisfying rule 2-3 can be found
            contrastive_score = torch.einsum(
                "id, jd->ij",
                batched_resume_coarse_vec / self.args.temperature,
                batched_job_coarse_vec,
            )

            bsz = batched_resume_coarse_vec.shape[0]
            labels = torch.arange(
                0, bsz, dtype=torch.long, device=contrastive_score.device
            )
            contrastive_loss += torch.nn.functional.cross_entropy(
                contrastive_score, labels
            )

        if len(batched_resume_hard_negatives) > 0:
            hard_neg_resume_output = self.forward(
                batched_resume_hard_negatives, data_type="resume"
            )
            batched_resume_hard_negatives_coarse_vec = hard_neg_resume_output.coarse_embedding
            batched_resume_hard_negatives_finegrained_vec = hard_neg_resume_output.finegrained_embedding

            batched_all_coarse_resume = torch.cat(
                [batched_resume_coarse_vec, batched_resume_hard_negatives_coarse_vec],
                dim=0,
            )
            contrastive_score = torch.einsum(
                "id, jd->ij",
                batched_job_coarse_vec / self.args.temperature,
                batched_all_coarse_resume,
            )

            # coarse level loss
            bsz = batched_job_coarse_vec.shape[0]
            labels = torch.arange(
                0, bsz, dtype=torch.long, device=contrastive_score.device
            )
            contrastive_loss += torch.nn.functional.cross_entropy(
                contrastive_score, labels
            )

            num_resume_hard_negs_used += batched_resume_hard_negatives_coarse_vec.shape[0]
        
        if len(batched_job_hard_negatives) > 0:
            # learn resume-to-job pairing since we have hard negatives OF job
            hard_neg_job_output = self.forward(
                batched_job_hard_negatives, data_type="job"
            )
            batched_job_hard_negatives_coarse_vec = hard_neg_job_output.coarse_embedding
            batched_job_hard_negatives_finegrained_vec = hard_neg_job_output.finegrained_embedding

            batched_all_coarse_job = torch.cat(
                [batched_job_coarse_vec, batched_job_hard_negatives_coarse_vec], dim=0
            )
            contrastive_score = torch.einsum(
                "id, jd->ij",
                batched_resume_coarse_vec / self.args.temperature,
                batched_all_coarse_job,
            )

            # coarse level loss
            bsz = batched_resume_coarse_vec.shape[0]
            labels = torch.arange(
                0, bsz, dtype=torch.long, device=contrastive_score.device
            )
            contrastive_loss += torch.nn.functional.cross_entropy(
                contrastive_score, labels
            )
            num_job_hard_negs_used += batched_job_hard_negatives_coarse_vec.shape[
                0
            ]
        return {
            "contrastive_loss": contrastive_loss,
            "num_resume_hard_negs_used": num_resume_hard_negs_used,
            "num_job_hard_negs_used": num_job_hard_negs_used,
            "num_resume_positive_used": num_resume_positive_used,
            "num_job_positive_used": num_job_positive_used,
            "batched_job_hard_negatives_coarse_vec": batched_job_hard_negatives_coarse_vec,
            "batched_job_hard_negatives_finegrained_vec": batched_job_hard_negatives_finegrained_vec,
            "batched_all_coarse_job": batched_all_coarse_job,
            "batched_resume_hard_negatives_coarse_vec": batched_resume_hard_negatives_coarse_vec,
            "batched_resume_hard_negatives_finegrained_vec": batched_resume_hard_negatives_finegrained_vec,
            "batched_all_coarse_resume": batched_all_coarse_resume,
        }

    def training_step(self, batch, batch_idx) -> torch.Tensor:
        """batch still consist of batched resume/job in the key dimension, BUT we have four differences:
        1. added hard negative, EITHER there is more resume than job, OR more job than resume
        2. the i-th resume is matched with the i-th job, i.e. the diagonal is the positive sample
        3. the hard negatives SHOULD NOT be a match to ANY of the batch_resume/batch_job (this is also used to calculate f1)
        4. (say we have hard negatives of resumes) batched_hard_negatives_indices specifies which pair of job each neg resume is matched with
        As a result, there is no label. the matching is done in dataset construction
        """
        batched_resume = batch["batched_resume"]
        batched_job = batch["batched_job"]
        batched_resume_hard_negatives = batch.get("batched_resume_hard_negatives", {})
        batched_job_hard_negatives = batch.get("batched_job_hard_negatives", {})
        batched_resume_hard_negatives_indices = batch.get("batched_resume_hard_negatives_indices", [])
        batched_job_hard_negatives_indices = batch.get("batched_job_hard_negatives_indices", [])
        neg_resume_only = batch["neg_resume_only"]  # back compatibility
        if not self.args.do_both_rj_hard_neg:
            if neg_resume_only:
                batched_job_hard_negatives = {}
            else:
                batched_resume_hard_negatives = {}
        # for rr-jj pair contrastive learning
        # indices indicates which row of resume/job this each positive sample is matched with
        batched_resume_positives = batch.get("batched_resume_positives", {})
        batched_resume_positive_indices = batch.get("batched_resume_positive_indices", [])
        batched_job_positives = batch.get("batched_job_positives", {})
        batched_job_positive_indices = batch.get("batched_job_positive_indices", [])

        ## compute embeddings
        resume_output = self.forward(batched_resume, data_type="resume")
        batched_resume_coarse_vec = resume_output.coarse_embedding
        batched_resume_finegrained_vec = resume_output.finegrained_embedding
        job_output = self.forward(batched_job, data_type="job")
        batched_job_coarse_vec = job_output.coarse_embedding
        batched_job_finegrained_vec = job_output.finegrained_embedding

        ## compute contrastive loss for coarse-grained embedding
        coarse_compute_output = self.compute_coarsegrained_loss(
            batched_resume_coarse_vec,
            batched_job_coarse_vec,
            batched_resume_hard_negatives,
            batched_resume_hard_negatives_indices,
            batched_job_hard_negatives,
            batched_job_hard_negatives_indices,
            batched_resume_positives,
            batched_resume_positive_indices,
            batched_job_positives,
            batched_job_positive_indices,
        )
        contrastive_loss = coarse_compute_output["contrastive_loss"]
        num_resume_hard_negs_used = coarse_compute_output["num_resume_hard_negs_used"]
        num_job_hard_negs_used = coarse_compute_output["num_job_hard_negs_used"]
        num_resume_positive_used = coarse_compute_output["num_resume_positive_used"]
        num_job_positive_used = coarse_compute_output["num_job_positive_used"]
        batched_job_hard_negatives_coarse_vec = coarse_compute_output[
            "batched_job_hard_negatives_coarse_vec"
        ]
        batched_job_hard_negatives_finegrained_vec = coarse_compute_output[
            "batched_job_hard_negatives_finegrained_vec"
        ]
        batched_all_coarse_job = coarse_compute_output["batched_all_coarse_job"]
        batched_resume_hard_negatives_coarse_vec = coarse_compute_output[
            "batched_resume_hard_negatives_coarse_vec"
        ]
        batched_resume_hard_negatives_finegrained_vec = coarse_compute_output[
            "batched_resume_hard_negatives_finegrained_vec"
        ]
        batched_all_coarse_resume = coarse_compute_output["batched_all_coarse_resume"]

        finegrained_loss = torch.zeros(
            1, dtype=torch.float, device=contrastive_loss.device
        )
        # per documentation for global_step: The number of OPTIMIZER steps taken (does not reset each epoch).
        trainer_actual_steps = (
            self.trainer.global_step * self.args.gradient_accumulation_steps
        )
        if (
            trainer_actual_steps
            > self.args.coarse_training_portion * self.trainer.max_steps
        ):
            # we have resume hard negs, then compute pairing of job with resume hard negs
            # TODO: not used for now
            finegrained_loss = self.compute_finegrained_loss(
                batched_resume_hard_negatives_indices,
                batched_job_hard_negatives_indices,
                batched_resume_coarse_vec,
                batched_resume_finegrained_vec,
                batched_job_coarse_vec,
                batched_job_finegrained_vec,
                batched_job_hard_negatives_coarse_vec=batched_job_hard_negatives_coarse_vec,
                batched_job_hard_negatives_finegrained_vec=batched_job_hard_negatives_finegrained_vec,
                batched_all_coarse_job=batched_all_coarse_job,
                batched_resume_hard_negatives_coarse_vec=batched_resume_hard_negatives_coarse_vec,
                batched_resume_hard_negatives_finegrained_vec=batched_resume_hard_negatives_finegrained_vec,
                batched_all_coarse_resume=batched_all_coarse_resume,
            )

        loss = (
            contrastive_loss
            + self.args.alpha * finegrained_loss
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
            self.log("train/contrastive_loss", contrastive_loss)
            self.log("train/finegrained_loss", finegrained_loss)
            self.log(
                "train/percent_resume_hard_neg",
                num_resume_hard_negs_used / batched_resume_coarse_vec.shape[0],
            )
            self.log(
                "train/percent_job_hard_neg",
                num_job_hard_negs_used / batched_job_coarse_vec.shape[0],
            )
            self.log(
                "train/percent_total_hard_neg",
                (num_resume_hard_negs_used + num_job_hard_negs_used)
                / batched_job_coarse_vec.shape[0],
            )
            self.log(
                "train/percent_resume_positive",
                num_resume_positive_used / batched_resume_coarse_vec.shape[0],
            )
            self.log(
                "train/percent_job_positive",
                num_job_positive_used / batched_job_coarse_vec.shape[0],
            )
            self.log(
                "train/percent_total_positive",
                (num_resume_positive_used + num_job_positive_used)
                / batched_job_coarse_vec.shape[0],
            )

            if self.args.finegrained_loss == "noop":
                prediction_resume_vecs = batched_resume_coarse_vec
                prediction_job_vecs = batched_job_coarse_vec
                prediction_resume_hard_neg_vec = batched_resume_hard_negatives_coarse_vec
                prediction_job_hard_neg_vec = batched_job_hard_negatives_coarse_vec
            else:
                prediction_resume_vecs = batched_resume_finegrained_vec
                prediction_job_vecs = batched_job_finegrained_vec
                prediction_resume_hard_neg_vec = batched_resume_hard_negatives_finegrained_vec
                prediction_job_hard_neg_vec = batched_job_hard_negatives_finegrained_vec

            # check the matching degree of training data
            cos_sim = torch.nn.CosineSimilarity(dim=0, eps=1e-6)
            score = cos_sim(prediction_resume_vecs[0], prediction_job_vecs[0])
            example_positive_prob = (score + 1) / 2.0
            self.log("train/example_positive_prob", example_positive_prob)
            if prediction_resume_hard_neg_vec is not None:
                hard_resume_paired_job = prediction_job_vecs[batched_resume_hard_negatives_indices]
                score = cos_sim(prediction_resume_hard_neg_vec[0], hard_resume_paired_job[0])
                example_negative_prob_hard_resume = (score + 1) / 2.0
                self.log("train/example_negative_prob_hard_resume", example_negative_prob_hard_resume)
            if prediction_job_hard_neg_vec is not None:
                hard_job_paired_resume = prediction_resume_vecs[batched_job_hard_negatives_indices]
                score = cos_sim(prediction_job_hard_neg_vec[0], hard_job_paired_resume[0])
                example_negative_prob_hard_job = (score + 1) / 2.0
                self.log("train/example_negative_prob_hard_job", example_negative_prob_hard_job)
            if prediction_job_vecs.shape[0] > 1:
                random_score = cos_sim(prediction_resume_vecs[0], prediction_job_vecs[1])
                example_random_prob = (random_score + 1) / 2.0
                self.log("train/example_random_prob", example_random_prob)

            geek_vec_length = torch.norm(prediction_resume_vecs, dim=1)
            geek_vec_avg_length = torch.mean(geek_vec_length)
            job_vec_length = torch.norm(prediction_job_vecs, dim=1)
            job_vec_avg_length = torch.mean(job_vec_length)
            self.log("train/geek_vec_avg_length", geek_vec_avg_length)
            self.log("train/job_vec_avg_length", job_vec_avg_length)

            num_positive = prediction_resume_vecs.shape[0]
            positive_score = torch.sum(
                prediction_resume_vecs * prediction_job_vecs, axis=1
            )
            normalized_positive_score = torch.sigmoid(positive_score)
            positive_class_preds = normalized_positive_score > 0.5

            # negative samples
            if num_resume_hard_negs_used > 0:
                num_negative = (
                    num_resume_hard_negs_used * prediction_job_vecs.shape[0]
                )
                negative_score = torch.einsum(
                    "id, jd->ij",
                    prediction_job_vecs,
                    prediction_resume_hard_neg_vec,
                )
                negative_score = torch.flatten(negative_score)  # all negative
                normalized_negative_score = torch.sigmoid(negative_score)
                negative_class_preds = normalized_negative_score > 0.5
            elif num_job_hard_negs_used > 0:
                num_negative = (
                    num_job_hard_negs_used * prediction_resume_vecs.shape[0]
                )
                negative_score = torch.einsum(
                    "id, jd->ij",
                    prediction_resume_vecs,
                    prediction_job_hard_neg_vec,
                )
                negative_score = torch.flatten(negative_score)
                normalized_negative_score = torch.sigmoid(negative_score)
                negative_class_preds = normalized_negative_score > 0.5
            else:
                # just add a dummy negative
                num_negative = 1
                negative_class_preds = torch.zeros(
                    num_negative,
                    dtype=torch.bool,
                    device=prediction_resume_vecs.device,
                )

            all_labels = torch.cat(
                [torch.ones(num_positive), torch.zeros(num_negative)], dim=0
            )
            all_preds = torch.cat([positive_class_preds, negative_class_preds], dim=0)
            self._log_metrics(all_preds, all_labels)
        return loss

    def validation_step(self, batch, batch_idx):
        batched_resume = batch["batched_resume"]
        batched_job = batch["batched_job"]
        batched_resume_hard_negatives = batch.get("batched_resume_hard_negatives", {})
        batched_job_hard_negatives = batch.get("batched_job_hard_negatives", {})
        batched_resume_hard_negatives_indices = batch.get("batched_resume_hard_negatives_indices", [])
        batched_job_hard_negatives_indices = batch.get("batched_job_hard_negatives_indices", [])
        neg_resume_only = batch["neg_resume_only"]  # back compatibility
        if not self.args.do_both_rj_hard_neg:
            if neg_resume_only:
                batched_job_hard_negatives = {}
            else:
                batched_resume_hard_negatives = {}
        # for rr-jj pair contrastive learning
        # indices indicates which row of resume/job this each positive sample is matched with
        batched_resume_positives = batch.get("batched_resume_positives", {})
        batched_resume_positive_indices = batch.get("batched_resume_positive_indices", [])
        batched_job_positives = batch.get("batched_job_positives", {})
        batched_job_positive_indices = batch.get("batched_job_positive_indices", [])

        ## compute embeddings
        resume_output = self.forward(batched_resume, data_type="resume")
        batched_resume_coarse_vec = resume_output.coarse_embedding
        batched_resume_finegrained_vec = resume_output.finegrained_embedding
        job_output = self.forward(batched_job, data_type="job")
        batched_job_coarse_vec = job_output.coarse_embedding
        batched_job_finegrained_vec = job_output.finegrained_embedding

        ## compute contrastive loss for coarse-grained embedding
        coarse_compute_output = self.compute_coarsegrained_loss(
            batched_resume_coarse_vec,
            batched_job_coarse_vec,
            batched_resume_hard_negatives,
            batched_resume_hard_negatives_indices,
            batched_job_hard_negatives,
            batched_job_hard_negatives_indices,
            batched_resume_positives,
            batched_resume_positive_indices,
            batched_job_positives,
            batched_job_positive_indices,
        )
        contrastive_loss = coarse_compute_output["contrastive_loss"]
        num_resume_hard_negs_used = coarse_compute_output["num_resume_hard_negs_used"]
        num_job_hard_negs_used = coarse_compute_output["num_job_hard_negs_used"]
        batched_job_hard_negatives_coarse_vec = coarse_compute_output[
            "batched_job_hard_negatives_coarse_vec"
        ]
        batched_job_hard_negatives_finegrained_vec = coarse_compute_output[
            "batched_job_hard_negatives_finegrained_vec"
        ]
        batched_all_coarse_job = coarse_compute_output["batched_all_coarse_job"]
        batched_resume_hard_negatives_coarse_vec = coarse_compute_output[
            "batched_resume_hard_negatives_coarse_vec"
        ]
        batched_resume_hard_negatives_finegrained_vec = coarse_compute_output[
            "batched_resume_hard_negatives_finegrained_vec"
        ]
        batched_all_coarse_resume = coarse_compute_output["batched_all_coarse_resume"]

        finegrained_loss = self.compute_finegrained_loss(
            batched_resume_hard_negatives_indices,
            batched_job_hard_negatives_indices,
            batched_resume_coarse_vec,
            batched_resume_finegrained_vec,
            batched_job_coarse_vec,
            batched_job_finegrained_vec,
            batched_job_hard_negatives_coarse_vec=batched_job_hard_negatives_coarse_vec,
            batched_job_hard_negatives_finegrained_vec=batched_job_hard_negatives_finegrained_vec,
            batched_all_coarse_job=batched_all_coarse_job,
            batched_resume_hard_negatives_coarse_vec=batched_resume_hard_negatives_coarse_vec,
            batched_resume_hard_negatives_finegrained_vec=batched_resume_hard_negatives_finegrained_vec,
            batched_all_coarse_resume=batched_all_coarse_resume,
        )

        loss = (
            contrastive_loss
            + self.args.alpha * finegrained_loss
        )

        # logging to tensorboard by default.
        # You will need to setup a folder `lightning_logs` in advance
        # also used to print to console
        self.log("val/loss", loss, sync_dist=True)
        self.log("val/contrastive_loss", contrastive_loss, sync_dist=True)
        self.log("val/finegrained_loss", finegrained_loss, sync_dist=True)

        if self.args.finegrained_loss == "noop":
            prediction_resume_vecs = batched_resume_coarse_vec
            prediction_job_vecs = batched_job_coarse_vec
            prediction_resume_hard_neg_vec = batched_resume_hard_negatives_coarse_vec
            prediction_job_hard_neg_vec = batched_job_hard_negatives_coarse_vec
        else:
            prediction_resume_vecs = batched_resume_finegrained_vec
            prediction_job_vecs = batched_job_finegrained_vec
            prediction_resume_hard_neg_vec = batched_resume_hard_negatives_finegrained_vec
            prediction_job_hard_neg_vec = batched_job_hard_negatives_finegrained_vec

        # check the matching degree of training data
        cos_sim = torch.nn.CosineSimilarity(dim=0, eps=1e-6)
        score = cos_sim(prediction_resume_vecs[0], prediction_job_vecs[0])
        example_positive_prob = (score + 1) / 2.0
        self.log("val/example_positive_prob", example_positive_prob, sync_dist=True)
        if prediction_resume_hard_neg_vec is not None:
            hard_resume_paired_job = prediction_job_vecs[batched_resume_hard_negatives_indices]
            score = cos_sim(prediction_resume_hard_neg_vec[0], hard_resume_paired_job[0])
            example_negative_prob_hard_resume = (score + 1) / 2.0
            self.log("val/example_negative_prob_hard_resume", example_negative_prob_hard_resume, sync_dist=True)
        if prediction_job_hard_neg_vec is not None:
            hard_job_paired_resume = prediction_resume_vecs[batched_job_hard_negatives_indices]
            score = cos_sim(prediction_job_hard_neg_vec[0], hard_job_paired_resume[0])
            example_negative_prob_hard_job = (score + 1) / 2.0
            self.log("val/example_negative_prob_hard_job", example_negative_prob_hard_job, sync_dist=True)
        if prediction_job_vecs.shape[0] > 1:
            random_score = cos_sim(prediction_resume_vecs[0], prediction_job_vecs[1])
            example_random_prob = (random_score + 1) / 2.0
            self.log("val/example_random_prob", example_random_prob, sync_dist=True)
        
        num_positive = prediction_resume_vecs.shape[0]
        positive_score = torch.sum(
            prediction_resume_vecs * prediction_job_vecs, axis=1
        )
        normalized_positive_score = torch.sigmoid(positive_score)
        positive_class_preds = normalized_positive_score > 0.5

        # negative samples
        if num_resume_hard_negs_used > 0:
            num_negative = (
                num_resume_hard_negs_used * prediction_job_vecs.shape[0]
            )
            negative_score = torch.einsum(
                "id, jd->ij",
                prediction_job_vecs,
                prediction_resume_hard_neg_vec,
            )
            negative_score = torch.flatten(negative_score)  # all negative
            normalized_negative_score = torch.sigmoid(negative_score)
            negative_class_preds = normalized_negative_score > 0.5
        elif num_job_hard_negs_used > 0:
            num_negative = (
                num_job_hard_negs_used * prediction_resume_vecs.shape[0]
            )
            negative_score = torch.einsum(
                "id, jd->ij",
                prediction_resume_vecs,
                prediction_job_hard_neg_vec,
            )
            negative_score = torch.flatten(negative_score)
            normalized_negative_score = torch.sigmoid(negative_score)
            negative_class_preds = normalized_negative_score > 0.5
        else:
            # just add a dummy negative
            num_negative = 1
            negative_class_preds = torch.zeros(
                num_negative,
                dtype=torch.bool,
                device=prediction_resume_vecs.device,
            )

        all_labels = torch.cat(
            [torch.ones(num_positive), torch.zeros(num_negative)], dim=0
        )
        all_preds = torch.cat([positive_class_preds, negative_class_preds], dim=0)
        self._log_metrics(all_preds, all_labels, mode="val")
        return loss

    def predict_step(self, batch, batch_idx):
        # assumes just batched_resume and batched_job
        batched_resume = batch["batched_resume"]
        batched_job = batch["batched_job"]

        resume_output = self.forward(batched_resume, data_type="resume")
        batched_resume_coarse_vec = resume_output.coarse_embedding
        batched_resume_finegrained_vec = resume_output.finegrained_embedding

        job_output = self.forward(batched_job, data_type="job")
        batched_job_coarse_vec = job_output.coarse_embedding
        batched_job_finegrained_vec = job_output.finegrained_embedding

        if self.args.finegrained_loss == "noop":
            score = torch.sum(
                batched_resume_coarse_vec * batched_job_coarse_vec, axis=1
            )
            return ModelOutput(
                output=score,
                batched_resume_representation=batched_resume_coarse_vec,
                batched_job_representation=batched_job_coarse_vec,
                batched_resume_coarse_representation=batched_resume_coarse_vec,
                batched_job_coarse_representation=batched_job_coarse_vec,
            )
        else:
            score = torch.sum(
                batched_resume_finegrained_vec * batched_job_finegrained_vec, axis=1
            )
            return ModelOutput(
                output=score,
                batched_resume_representation=batched_resume_finegrained_vec,
                batched_job_representation=batched_job_finegrained_vec,
                batched_resume_coarse_representation=batched_resume_coarse_vec,
                batched_job_coarse_representation=batched_job_coarse_vec,
            )

    def configure_optimizers(self):
        if self.args.cpu_adam:
            optimizer = DeepSpeedCPUAdam(
                self.parameters(),
                lr=self.args.lr,
                weight_decay=self.args.weight_decay,
            )
        else:
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

    def load_state_dict(self, state_dict, strict=True):
        """issue can arise if deepspeed only saves one copy of the weights when shared_encoder=True.
        Technically this is great, but the default load_state_dict call will throw error as it cannot
        find the weights. Let's just manually do it here.

        Args:
            state_dict (_type_): _description_
            strict (bool, optional): _description_. Defaults to True.
        """
        # the main weights is from bert_resume when its shared_encoder=True
        # we need to manually load the weights to bert_job
        if "bert_job" not in state_dict and self.args.share_encoder:
            incompatible_keys = super().load_state_dict(state_dict, strict=False)
            missing_keys = incompatible_keys.missing_keys
            real_missing_keys = []
            for key in missing_keys:
                if "bert_job" in key:
                    continue
                real_missing_keys.append(key)
            assert len(real_missing_keys) == 0, f"missing keys {real_missing_keys}"
            self.bert_job = self.bert_resume
        else:
            # let super handle the rest
            return super().load_state_dict(state_dict, strict=strict)
        return
