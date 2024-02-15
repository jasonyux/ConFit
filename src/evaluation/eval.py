from typing import List, Tuple, Dict
from src.evaluation.metrics import Metric
from sklearn.metrics import (
    f1_score, confusion_matrix, roc_auc_score,
    ndcg_score,
)
import numpy as np


class EvalFindClassificationThreshold:
    """same input as EvalClassification, but finds the best threshold for classification
    """
    def __init__(self,
        metric,
        test_rid_to_representation: dict,
        test_jid_to_representation: dict,
        test_pairs: List[Tuple[str, str, int]],
        offline_mode: bool = False,
    ):
        self.metric = metric
        self.test_rid_to_representation = test_rid_to_representation
        self.test_jid_to_representation = test_jid_to_representation
        self.test_pairs = test_pairs
        self.offline_mode = offline_mode  # if offline_mode=True, assume that the metric already contains precomputed scores
        return
    
    def _prepare_online_data(self):
        all_to_test_resume = []
        all_to_test_job = []
        all_to_test_label = []
        for user_id, jd_no, label in self.test_pairs:
            all_to_test_resume.append(self.test_rid_to_representation[user_id])
            all_to_test_job.append(self.test_jid_to_representation[jd_no])
            all_to_test_label.append(label)

        all_to_test_resume = np.array(all_to_test_resume)
        all_to_test_job = np.array(all_to_test_job)
        all_to_test_label = np.array(all_to_test_label)
        return all_to_test_resume, all_to_test_job, all_to_test_label

    def _prepare_offline_data(self):
        all_to_test_resume = []
        all_to_test_job = []
        all_to_test_label = []
        for user_id, jd_no, label in self.test_pairs:
            all_to_test_resume.append(user_id)
            all_to_test_job.append(jd_no)
            all_to_test_label.append(label)
        return all_to_test_resume, all_to_test_job, all_to_test_label

    def search_best_threshold(self, scores, labels):
        all_thresholds = scores.copy()
        all_thresholds.sort()
        all_thresholds = np.unique(all_thresholds)

        all_scores = []
        for threshold in all_thresholds:
            preds = scores > threshold
            f1 = f1_score(labels, preds, average="weighted")
            all_scores.append(f1)

        all_scores = np.array(all_scores)
        np.nan_to_num(all_scores, 0)
        ix = np.argmax(all_scores)
        best_threshold = all_thresholds[ix]
        return float(best_threshold)

    def evaluate(self):
        if self.offline_mode:
            all_to_test_resume, all_to_test_job, all_to_test_label = self._prepare_offline_data()
        else:
            all_to_test_resume, all_to_test_job, all_to_test_label = self._prepare_online_data()

        scores = self.metric.batch_score(all_to_test_resume, all_to_test_job)
        
        # balanced sample weight
        best_threshold = self.search_best_threshold(scores, all_to_test_label)

        # calculate scores
        preds = scores > best_threshold
        roc_auc = roc_auc_score(all_to_test_label, scores, average="weighted")
        f1 = f1_score(all_to_test_label, preds, average="weighted")
        matrix = confusion_matrix(all_to_test_label, preds)
        precision_class_1 = matrix[1][1] / (matrix[1][1] + matrix[0][1])
        recall_class_1 = matrix[1][1] / (matrix[1][1] + matrix[1][0])

        score_report = {
            "best_threshold": best_threshold,
            "f1": f1,
            "roc_auc": roc_auc,
            "precision_class_1": precision_class_1,
            "recall_class_1": recall_class_1,
        }

        # history
        eval_history = []
        for i in range(len(all_to_test_label)):
            eval_history.append(
                {
                    "user_id": self.test_pairs[i][0],
                    "jd_no": self.test_pairs[i][1],
                    "label": self.test_pairs[i][2],
                    "score": scores[i],
                    "pred": preds[i],
                }
            )
        return score_report, eval_history


class EvalClassification:
    def __init__(
        self,
        metric: Metric,
        test_rid_to_representation: dict,
        test_jid_to_representation: dict,
        test_pairs: List[Tuple[str, str, int]],
        threshold: float = 0.5,
        offline_mode: bool = False,
    ):
        self.metric = metric
        self.test_rid_to_representation = test_rid_to_representation
        self.test_jid_to_representation = test_jid_to_representation
        self.test_pairs = test_pairs
        self.threshold = threshold
        self.offline_mode = offline_mode  # if offline_mode=True, assume that the metric already contains precomputed scores
        return

    def _prepare_online_data(self):
        all_to_test_resume = []
        all_to_test_job = []
        all_to_test_label = []
        for user_id, jd_no, label in self.test_pairs:
            all_to_test_resume.append(self.test_rid_to_representation[user_id])
            all_to_test_job.append(self.test_jid_to_representation[jd_no])
            all_to_test_label.append(label)

        all_to_test_resume = np.array(all_to_test_resume)
        all_to_test_job = np.array(all_to_test_job)
        all_to_test_label = np.array(all_to_test_label)
        return all_to_test_resume, all_to_test_job, all_to_test_label

    def _prepare_offline_data(self):
        all_to_test_resume = []
        all_to_test_job = []
        all_to_test_label = []
        for user_id, jd_no, label in self.test_pairs:
            all_to_test_resume.append(user_id)
            all_to_test_job.append(jd_no)
            all_to_test_label.append(label)
        return all_to_test_resume, all_to_test_job, all_to_test_label

    def evaluate(self):
        if self.offline_mode:
            all_to_test_resume, all_to_test_job, all_to_test_label = self._prepare_offline_data()
        else:
            all_to_test_resume, all_to_test_job, all_to_test_label = self._prepare_online_data()

        scores = self.metric.batch_score(all_to_test_resume, all_to_test_job)
        preds = scores > self.threshold

        # calculate scores
        roc_auc = roc_auc_score(all_to_test_label, scores)
        f1 = f1_score(all_to_test_label, preds, average="weighted")
        matrix = confusion_matrix(all_to_test_label, preds)
        precision_class_1 = matrix[1][1] / (matrix[1][1] + matrix[0][1])
        recall_class_1 = matrix[1][1] / (matrix[1][1] + matrix[1][0])

        score_report = {
            "threshold": self.threshold,
            "f1": f1,
            "roc_auc": roc_auc,
            "precision_class_1": precision_class_1,
            "recall_class_1": recall_class_1,
        }

        # history
        eval_history = []
        for i in range(len(all_to_test_label)):
            eval_history.append(
                {
                    "user_id": self.test_pairs[i][0],
                    "jd_no": self.test_pairs[i][1],
                    "label": self.test_pairs[i][2],
                    "score": scores[i],
                    "pred": preds[i],
                }
            )
        return score_report, eval_history


class EvalRanking:
    def __init__(
        self,
        metric: Metric,
        test_rid_to_representation: dict,
        test_jid_to_representation: dict,
        test_ranking_data: Dict[str, List[str]],
        offline_mode: bool = False,
    ):
        self.metric = metric
        self.test_rid_to_representation = test_rid_to_representation
        self.test_jid_to_representation = test_jid_to_representation
        self.test_ranking_data = test_ranking_data
        self.offline_mode = offline_mode
        return

    def ranking_mode(self):
        for _, v in self.test_ranking_data.items():
            if "jd_nos" in v:
                return "rank_job"
            elif "user_ids" in v:
                return "rank_user"
            else:
                raise Exception("unknown ranking mode")
        return

    def _calculate_ap(self, predicted_ranking, gold_labels: dict):
        """
        predicted_ranking: i-th entry stores the the index of the resume/job that is ranked i-th
        gold_labels: stores whether if the i-th job-resume pair is satisfied
        """
        # calculate Average Precision
        num_positive_seen = 0
        ap = 0
        for i, j in enumerate(predicted_ranking):
            if gold_labels[j] == 1:
                num_positive_seen += 1
                ap += num_positive_seen / (i + 1)  # (i+1 is the index/predicted rank)
        ap /= num_positive_seen
        return ap

    def _calculate_ndcg(self, scores: np.ndarray, gold_labels: np.ndarray, k=None):
        gold_labels = gold_labels > 0
        true_relevance = np.asarray([gold_labels], dtype=np.int32)
        scores = np.asarray([scores])
        ndcg = ndcg_score(true_relevance, scores, k=k)
        return ndcg

    def evaluate(self):
        ranking_mode = self.ranking_mode()
        all_ap_scores = []
        all_ndcg_scores = []
        all_ndcg_10_scores = []
        eval_history = []
        for k, v in self.test_ranking_data.items():
            k = str(k)
            if ranking_mode == "rank_job":
                user_id = k
                jd_nos = v["jd_nos"]
                labels = v["satisfied"]
                
                if self.offline_mode:
                    scores = self.metric.batch_score(
                        np.array([user_id] * len(jd_nos)),
                        np.array(
                            [str(jd_no) for jd_no in jd_nos]
                        ),
                    )
                else:
                    scores = self.metric.batch_score(
                        np.array([self.test_rid_to_representation[user_id]] * len(jd_nos)),
                        np.array(
                            [self.test_jid_to_representation[jd_no] for jd_no in jd_nos]
                        ),
                    )

                predicted_ranking = np.argsort(scores)[::-1]
                ap = self._calculate_ap(predicted_ranking, labels)
                ndcg = self._calculate_ndcg(scores, np.array(labels))
                ndcg_10 = self._calculate_ndcg(scores, np.array(labels), k=10)

                eval_history.append(
                    {
                        "user_id": user_id,
                        "jd_nos": jd_nos,
                        "labels": labels,
                        "scores": scores,
                        "predicted_ranking": predicted_ranking,
                    }
                )
                all_ap_scores.append(ap)
                all_ndcg_scores.append(ndcg)
                all_ndcg_10_scores.append(ndcg_10)
            elif ranking_mode == "rank_user":
                jd_no = k
                user_ids = v["user_ids"]
                labels = v["satisfied"]
                if self.offline_mode:
                    scores = self.metric.batch_score(
                        np.array(
                            [
                                str(user_id)
                                for user_id in user_ids
                            ]
                        ),
                        np.array([jd_no] * len(user_ids)),
                    )
                else:
                    scores = self.metric.batch_score(
                        np.array(
                            [
                                self.test_rid_to_representation[user_id]
                                for user_id in user_ids
                            ]
                        ),
                        np.array([self.test_jid_to_representation[jd_no]] * len(user_ids)),
                    )

                predicted_ranking = np.argsort(scores)[::-1]
                ap = self._calculate_ap(predicted_ranking, labels)
                ndcg = self._calculate_ndcg(scores, np.array(labels))
                ndcg_10 = self._calculate_ndcg(scores, np.array(labels), k=10)

                eval_history.append(
                    {
                        "jd_no": jd_no,
                        "user_ids": user_ids,
                        "labels": labels,
                        "scores": scores,
                        "predicted_ranking": predicted_ranking,
                    }
                )
                all_ap_scores.append(ap)
                all_ndcg_scores.append(ndcg)
                all_ndcg_10_scores.append(ndcg_10)

        # calculate scores
        mean_ap = np.mean(all_ap_scores)
        mean_ndcg = np.mean(all_ndcg_scores)
        mean_ndcg_10 = np.mean(all_ndcg_10_scores)

        score_report = {
            "map": mean_ap,
            "ndcg": mean_ndcg,
            "ndcg@10": mean_ndcg_10,
        }
        return score_report, eval_history
