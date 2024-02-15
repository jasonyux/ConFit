from torchmetrics import Precision, Recall, F1Score
from transformers import AutoModel, BatchEncoding
from transformers.utils import ModelOutput
from transformers import get_linear_schedule_with_warmup
from dataclasses import dataclass, field
from src.model.base import BaseModel
import torch
import torch.nn as nn
import pytorch_lightning as pl


@dataclass
class MVCoNModelArguments:
    lr: float = field(default=5e-6)
    gradient_accumulation_steps: int = field(default=4)
    weight_decay: float = field(default=1e-2)
    num_warmup_steps: float = field(default=0.1)
    word_emb_dim: int = field(default=768)
    num_resume_features: int = field(
        default=12,
        metadata={"help": "Number of keys in resume"}
    )
    num_job_features: int = field(
        default=11,
        metadata={"help": "Number of keys in job description"}
    )
    pretrained_encoder: str = field(default="bert-base-chinese")


class TextCNN(nn.Module):
    def __init__(self, channels, kernel_size, pool_size, dim, method="max"):
        super(TextCNN, self).__init__()
        self.net1 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size[0]),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.MaxPool2d(pool_size),
        )
        self.net2 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size[1]),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.AdaptiveMaxPool2d((1, dim)),
        )
        if method == "max":
            self.pool = nn.AdaptiveMaxPool2d((1, dim))
        elif method == "mean":
            self.pool = nn.AdaptiveAvgPool2d((1, dim))
        else:
            raise ValueError("method {} not exist".format(method))

    def forward(self, x):
        # e.g. shape [b, num_features, seq_len, embed_dim]
        x = self.net1(x)
        x = self.net2(x).squeeze(2)
        x = self.pool(x).squeeze(1)
        return x


class MVCoN_Single(torch.nn.Module):
    def __init__(self, args: MVCoNModelArguments):
        super(MVCoN_Single, self).__init__()
        # dim = args["model"]["dim"]
        # self.emb = nn.Embedding(args["model"]["word_num"], dim, padding_idx=0)
        self.args = args
        self.emb = AutoModel.from_pretrained(args.pretrained_encoder)

        self.geek_layer = TextCNN(
            channels=self.args.num_resume_features,
            kernel_size=[(5, 1), (3, 1)],
            pool_size=(2, 1),
            dim=self.args.word_emb_dim,
            method="max",
        )

        self.job_layer = TextCNN(
            channels=self.args.num_job_features,
            kernel_size=[(5, 1), (5, 1)],
            pool_size=(2, 1),
            dim=self.args.word_emb_dim,
            method="mean",
        )

        self.mlp = nn.Sequential(
            nn.Linear(self.args.word_emb_dim, self.args.word_emb_dim),
            nn.ReLU(),
            nn.Linear(self.args.word_emb_dim, 1),
        )
        return

    def forward(self, geek_vec, job_vec):
        # # the original implementation with an Embedding(dim=100) layer gives:
        # # geek_vec.shape = torch.Size([b, num_features, seq_len])
        # geek_vec = self.emb(**geek_vec)  # torch.Size([b, num_features, seq_len, embed_dim])
        # job_vec = self.emb(**job_vec)
        bsz = geek_vec["input_ids"].shape[0]
        geek_vecs = []
        for i in range(bsz):
            # torch.Size([num_features, seq_len, embed_dim])
            one_geek = {
                "input_ids": geek_vec["input_ids"][i],
                "attention_mask": geek_vec["attention_mask"][i],
            }
            if "token_type_ids" in geek_vec:
                one_geek["token_type_ids"] = geek_vec["token_type_ids"][i]
            one_gee_repr = self.emb(**one_geek).last_hidden_state
            geek_vecs.append(one_gee_repr)
        geek_vecs = torch.stack(geek_vecs, dim=0)  # torch.Size([b, num_features, seq_len, embed_dim])

        job_vecs = []
        for i in range(bsz):
            one_job = {
                "input_ids": job_vec["input_ids"][i],
                "attention_mask": job_vec["attention_mask"][i],
            }
            if "token_type_ids" in job_vec:
                one_job["token_type_ids"] = job_vec["token_type_ids"][i]
            one_job_repr = self.emb(**one_job).last_hidden_state
            job_vecs.append(one_job_repr)
        job_vecs = torch.stack(job_vecs, dim=0)  # torch.Size([b, num_features, seq_len, embed_dim])

        # CNN
        geek_vecs = self.geek_layer(geek_vecs)
        job_vecs = self.job_layer(job_vecs)
        x = geek_vecs * job_vecs
        x = self.mlp(x).squeeze(1)
        return ModelOutput(
            logits=x,
            geek_vecs=geek_vecs,
            job_vecs=job_vecs,
        )


class MVCoN(BaseModel):
    def __init__(self, args: MVCoNModelArguments):
        super(MVCoN, self).__init__()
        self.args = args
        self.automatic_optimization = (
            False  # we manually do optimizer.step() and scheduler.step()
        )
        self.num_classes = 2

        self.encoder_1 = MVCoN_Single(args)
        self.encoder_2 = MVCoN_Single(args)

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

    def _chunk_batch_encoding(self, batch_encoding: BatchEncoding, chunks, dim):
        """
        Chunk a batch encoding into two halves.
        """
        chunked_1 = {}
        chunked_2 = {}
        for key, v in batch_encoding.items():
            # chunk along batch dimension
            if v.shape[dim] < chunks:
                # just repeat
                chunk_1_ = v
                chunk_2_ = v
            else:
                chunk_1_, chunk_2_ = torch.chunk(v, chunks, dim)
            chunked_1[key] = chunk_1_
            chunked_2[key] = chunk_2_
        return chunked_1, chunked_2

    def forward(self, resume_sents: BatchEncoding, job_sents: BatchEncoding, training=True):
        device = self.encoder_1.emb.device
        resume_sents = resume_sents.to(device)
        job_sents = job_sents.to(device)

        if training:
            geek_sent1, geek_sent2 = self._chunk_batch_encoding(resume_sents, 2, 0)
            job_sent1, job_sent2 = self._chunk_batch_encoding(job_sents, 2, 0)

            encoder_1_output_2 = self.encoder_1(geek_sent2, job_sent2).logits
            encoder_2_output_1 = self.encoder_2(geek_sent1, job_sent1).logits

            encoder_1_output_1 = self.encoder_1(geek_sent1, job_sent1).logits
            encoder_2_output_2 = self.encoder_2(geek_sent2, job_sent2).logits
            return ModelOutput(
                encoder_1_output_2=encoder_1_output_2,
                encoder_2_output_1=encoder_2_output_1,
                encoder_1_output_1=encoder_1_output_1,
                encoder_2_output_2=encoder_2_output_2,
            )
        else:
            encoder_1_output = self.encoder_1(resume_sents, job_sents).logits
            encoder_2_output = self.encoder_2(resume_sents, job_sents).logits
            return ModelOutput(
                encoder_1_output=encoder_1_output, encoder_2_output=encoder_2_output
            )

    def training_step(self, batch, batch_idx):
        device = self.encoder_1.emb.device
        batched_label = batch.pop("label")

        model_output = self(**batch)
        batched_label = batched_label.to(device)

        if batched_label.shape[0] < 2:
            # cannot be chunked, just repeat. 
            # Note that this is also done in self(**batch) for consistency
            labels1 = batched_label
            labels2 = batched_label
        else:
            labels1, labels2 = torch.chunk(batched_label, 2, 0)
        labels1 = labels1.float().to(device)
        labels2 = labels2.float().to(device)
        
        outputs2 = model_output.encoder_1_output_2
        outputs1 = model_output.encoder_2_output_1
        # criterion_weight = nn.BCEWithLogitsLoss(reduction = 'none')
        # loss_weight1 = criterion_weight(outputs1, labels1)
        loss_weight1 = torch.nn.functional.binary_cross_entropy_with_logits(
            outputs1, labels1, reduction="none"
        )
        # loss_weight2 = criterion_weight(outputs2, labels2)
        loss_weight2 = torch.nn.functional.binary_cross_entropy_with_logits(
            outputs2, labels2, reduction="none"
        )

        weight1 = torch.add(
            torch.ones(labels1.size()).float().to(device),
            0.2,
            torch.mul(
                torch.sub(labels1, torch.ones(labels1.size()).float().to(device)),
                loss_weight1,
            ),
        )
        weight2 = torch.add(
            torch.ones(labels2.size()).float().to(device),
            0.2,
            torch.mul(
                torch.sub(labels2, torch.ones(labels2.size()).float().to(device)),
                loss_weight2,
            ),
        )

        real_output_1 = model_output.encoder_1_output_1
        real_output_2 = model_output.encoder_2_output_2
        # loss1 = criterion(real_output_1, weight1.detach() * labels1)
        loss1 = torch.nn.functional.binary_cross_entropy_with_logits(
            real_output_1, weight1.detach() * labels1
        )
        # loss2 = criterion(real_output_2, weight2.detach() * labels2)
        loss2 = torch.nn.functional.binary_cross_entropy_with_logits(
            real_output_2, weight2.detach() * labels2
        )

        ### optimizer step.
        # Essential since scheduler from transformer does not directly work with pl lightning, we need to manually step
        optimizer = self.optimizers()
        loss1 /= self.args.gradient_accumulation_steps
        loss2 /= self.args.gradient_accumulation_steps
        self.manual_backward(loss1)
        self.manual_backward(loss2)
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
        train_loss = (loss1 + loss2) / 2.0
        self.log("train/loss", train_loss)
        with torch.no_grad():
            # class_preds = torch.argmax(pred, dim=1)
            class_preds_1 = torch.sigmoid(real_output_1) > 0.5
            class_preds_2 = torch.sigmoid(real_output_2) > 0.5
            class_preds = torch.cat([class_preds_1, class_preds_2], dim=0)
            batched_label = torch.cat([labels1, labels2], dim=0)
            self._log_metrics(class_preds, batched_label)
        return train_loss

    def validation_step(self, batch, batch_idx):
        # MV-CoN is cheating a bit by also doing this choice in test set
        device = self.encoder_1.emb.device
        batched_label = batch.pop("label")

        model_output = self(**batch, training=False)
        outputs1 = model_output.encoder_1_output
        outputs2 = model_output.encoder_2_output

        # check the loss
        batched_label = batched_label.float().to(device)
        loss1 = torch.nn.functional.binary_cross_entropy_with_logits(outputs1, batched_label)
        loss2 = torch.nn.functional.binary_cross_entropy_with_logits(outputs2, batched_label)
        val_loss = (loss1 + loss2) / 2.0

        self.log("val/loss", val_loss, sync_dist=True)

        # MV-CoN uses the second encoder for prediction
        class_preds = torch.sigmoid(outputs2) > 0.5
        self._log_metrics(class_preds, batched_label, mode="val")
        return val_loss

    def predict_step(self, batch, batch_idx):
        if "label" in batch:
            _ = batch.pop("label")
        
        # model_output = self(**batch)
        model_output = self(**batch, training=False)
        # MV-CoN uses the second encoder for prediction
        logits = model_output.encoder_2_output
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
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(self.args.num_warmup_steps * self.trainer.max_steps),
            num_training_steps=self.trainer.max_steps,
        )
        return [optimizer], [scheduler]
