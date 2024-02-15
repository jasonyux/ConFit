from typing import List, Union, Dict, Tuple
from xgboost import XGBClassifier, XGBRanker
from sklearn.metrics import ndcg_score
from sklearn.utils import class_weight
from sklearn.model_selection import ParameterGrid
from tqdm.auto import tqdm
from abc import ABC, abstractmethod
import numpy as np
import os
import json
import logging


logger = logging.getLogger(__name__)


class Metric(ABC):
    @abstractmethod
    def score(
        self,
        resume_repersentations: Union[np.ndarray, str],
        job_representations: Union[np.ndarray, str],
    ) -> float:
        """return probability of matching of this pair of resume and job

        Args:
            resume_repersentations (Union[np.ndarray, str]): either an embedding of that resume, or resume id (if you precomputed all scores)
            job_representations (Union[np.ndarray, str]): either an embedding of that job, or job id (if you precomputed all scores)
        
        Returns:
            float: probability of matching
        """
        raise NotImplementedError
    
    @abstractmethod
    def batch_score(
        self,
        resume_repersentations: Union[List[np.ndarray], List[str]],
        job_representations: Union[List[np.ndarray], List[str]],
    ) -> np.ndarray:
        raise NotImplementedError


class DecisionTreeMetric(Metric):
    """
    Given a job representation and a resume representation, predict whether the matching score using a decision tree.
    This metric is also trainable, as it uses a decision tree to predict the matching score.
    """
    def __init__(
        self,
        train_resume_representations,
        train_job_representations,
        train_labels,
        valid_resume_representations,
        valid_job_representations,
        valid_labels,
        merge_op="concat",
        do_sweep=True,
        model_save_path=None,
        **training_hparams
    ):
        super().__init__()
        self.merge_op = merge_op
        if merge_op == "concat":
            self.train_X = np.concatenate(
                [train_resume_representations, train_job_representations], axis=1
            )
            self.train_Y = np.array(train_labels)
            self.valid_X = np.concatenate(
                [valid_resume_representations, valid_job_representations], axis=1
            )
            self.valid_Y = np.array(valid_labels)
        else:
            raise NotImplementedError

        # DT metric needs to be trained
        if do_sweep:
            logger.info("ignoring training_hparams")
            self.model, swept_hyparams = self._sweep_and_train()
            training_hparams = swept_hyparams
        else:
            self.model = self._train(**training_hparams)
        self.training_hparams = training_hparams

        if model_save_path is not None:
            if not os.path.exists(model_save_path):
                os.makedirs(model_save_path)

            self.model.save_model(os.path.join(model_save_path, "metric_model.bin"))
            with open(
                os.path.join(model_save_path, "hyperparams.json"), "w", encoding="utf-8"
            ) as f:
                json.dump(training_hparams, f, indent=4)
        return

    def _train(self, **kwargs):
        np.random.seed(42)
        bst = XGBClassifier(**kwargs)
        # fit model
        classes_weights = class_weight.compute_sample_weight(
            class_weight="balanced", y=self.train_Y
        )
        bst.fit(
            self.train_X,
            self.train_Y,
            eval_set=[(self.train_X, self.train_Y), (self.valid_X, self.valid_Y)],
            sample_weight=classes_weights,
            verbose=True,
        )
        return bst

    def __train_and_eval(self, hyperparams) -> float:
        bst = XGBClassifier(**hyperparams)
        # fit model
        bst.fit(
            np.array(self.train_X),
            np.array(self.train_Y),
            eval_set=[(self.train_X, self.train_Y), (self.valid_X, self.valid_Y)],
            verbose=False,
        )

        pred_probs = bst.predict_proba(np.array(self.valid_X))
        preds = np.argmax(pred_probs, axis=1)
        num_correct = np.sum(preds == np.array(self.valid_Y))

        acc = num_correct / len(preds)
        return acc

    def _sweep(self):
        logger.info("sweeping hyperparams")
        np.random.seed(42)

        logger.info("sweeping stage 1")

        grid_group_1 = {
            "learning_rate": [1e-2],
            "n_estimators": [1000],
            "max_depth": [4, 6, 8, 10, 12],
            "min_child_weight": [1, 2, 4, 6],
            "subsample": [1],
            "colsample_bytree": [1],
            "tree_method": ["gpu_hist"],
            "random_state": [42],
            "early_stopping_rounds": [20],
        }

        all_scores = []
        param_grid = ParameterGrid(grid_group_1)
        for p in tqdm(param_grid, total=len(param_grid)):
            acc = self.__train_and_eval(p)
            all_scores.append((p, acc))

        best_param, best_score = sorted(all_scores, key=lambda x: x[1], reverse=True)[0]

        logger.info("stage 1 best param and score")
        logger.info(best_param)
        logger.info(best_score)

        np.random.seed(42)
        logger.info("stage 2")

        grid_group_2 = {
            "learning_rate": [1e-2],
            "n_estimators": [1000],
            "max_depth": [best_param["max_depth"]],
            "min_child_weight": [best_param["min_child_weight"]],
            "subsample": [1.0, 0.95, 0.9, 0.85],
            "colsample_bytree": [1.0, 0.95, 0.9, 0.85],
            "tree_method": ["gpu_hist"],
            "random_state": [42],
            "early_stopping_rounds": [20],
        }

        all_scores = []
        param_grid = ParameterGrid(grid_group_2)
        for p in tqdm(param_grid, total=len(param_grid)):
            acc = self.__train_and_eval(p)
            all_scores.append((p, acc))

        best_param, best_score = sorted(all_scores, key=lambda x: x[1], reverse=True)[0]

        logger.info("stage 2 best param and score")
        logger.info(best_param)

        np.random.seed(42)
        logger.info("stage 3")

        grid_group_3 = {
            "learning_rate": [1e-1, 5e-2, 1e-2, 5e-3, 1e-3],
            "n_estimators": [1000, 1200, 1500, 800, 600],
            "max_depth": [best_param["max_depth"]],
            "min_child_weight": [best_param["min_child_weight"]],
            "subsample": [best_param["subsample"]],
            "colsample_bytree": [best_param["colsample_bytree"]],
            "tree_method": ["gpu_hist"],
            "random_state": [42],
            "early_stopping_rounds": [20],
        }

        all_scores = []
        param_grid = ParameterGrid(grid_group_3)
        for p in tqdm(param_grid, total=len(param_grid)):
            acc = self.__train_and_eval(p)
            all_scores.append((p, acc))

        best_param, best_score = sorted(all_scores, key=lambda x: x[1], reverse=True)[0]
        logger.info("stage 3 best param and score")
        logger.info(best_param)
        return best_param

    def _sweep_and_train(self):
        best_param = self._sweep()
        model = self._train(**best_param)
        return model, best_param

    def batch_score(
        self,
        resume_repersentations: List[np.ndarray],
        job_representations: List[np.ndarray],
    ) -> np.ndarray:
        if self.merge_op == "concat":
            x = np.concatenate([resume_repersentations, job_representations], axis=1)
        else:
            raise NotImplementedError
        pred_probs = self.model.predict_proba(x)[:, 1]
        return pred_probs

    def score(
        self, resume_repersentation: np.ndarray, job_representation: np.ndarray
    ) -> float:
        prob = self.batch_score([resume_repersentation], [job_representation])[0]
        return float(prob)


class DecisionTreeRankerMetric(Metric):
    """
    Given a job representation and a resume representation, predict whether the matching score using a decision tree.
    This metric is also trainable, as it uses a decision tree to predict the matching score.
    """
    def __init__(
        self,
        train_resume_representations,
        train_job_representations,
        train_labels,
        train_group_ids,  # for Resume ranker, this would be the job ids of each pair
        valid_resume_representations,
        valid_job_representations,
        valid_labels,
        valid_group_ids,
        merge_op="concat",
        do_sweep=True,
        model_save_path=None,
        **training_hparams
    ):
        super().__init__()
        self.merge_op = merge_op
        if merge_op == "concat":
            self.train_X = np.concatenate(
                [train_resume_representations, train_job_representations], axis=1
            )
            self.train_Y = np.array(train_labels)
            self.train_group_ids = np.array(train_group_ids)
            self.valid_X = np.concatenate(
                [valid_resume_representations, valid_job_representations], axis=1
            )
            self.valid_Y = np.array(valid_labels)
            self.valid_group_ids = np.array(valid_group_ids)
        else:
            raise NotImplementedError

        # DT metric needs to be trained
        if do_sweep:
            logger.info("ignoring training_hparams")
            self.model, swept_hyparams = self._sweep_and_train()
            training_hparams = swept_hyparams
        else:
            self.model = self._train(**training_hparams)
        self.training_hparams = training_hparams

        if model_save_path is not None:
            if not os.path.exists(model_save_path):
                os.makedirs(model_save_path)

            self.model.save_model(os.path.join(model_save_path, "metric_model.bin"))
            with open(
                os.path.join(model_save_path, "hyperparams.json"), "w", encoding="utf-8"
            ) as f:
                json.dump(training_hparams, f, indent=4)
        return

    def _train(self, **kwargs):
        np.random.seed(42)
        bst = XGBRanker(**kwargs)
        # fit model
        bst.fit(
            self.train_X,
            self.train_Y,
            qid=self.train_group_ids,
            eval_set=[(self.train_X, self.train_Y), (self.valid_X, self.valid_Y)],
            eval_qid=[self.train_group_ids, self.valid_group_ids],
            verbose=True,
        )
        return bst

    def __train_and_eval(self, hyperparams) -> float:
        bst = XGBRanker(**hyperparams)
        # fit model
        bst.fit(
            np.array(self.train_X),
            np.array(self.train_Y),
            qid=self.train_group_ids,
            eval_set=[(self.train_X, self.train_Y), (self.valid_X, self.valid_Y)],
            eval_qid=[self.train_group_ids, self.valid_group_ids],
            verbose=False,
        )
        
        # compute ndcg
        all_ndcg = []
        group_by_gids = {}  # gid -> list of indices
        for i, gid in enumerate(self.valid_group_ids):
            if gid not in group_by_gids:
                group_by_gids[gid] = []
            group_by_gids[gid].append(i)

        for grouped_idx in group_by_gids.values():
            grouped_idx = np.array(grouped_idx)
            if len(grouped_idx) == 1:
                continue
            grouped_X = self.valid_X[grouped_idx]
            grouped_Y = self.valid_Y[grouped_idx].astype(np.int32)
            grouped_pred_scores = bst.predict(grouped_X)
            ndcg = ndcg_score(grouped_Y.reshape(1, -1), grouped_pred_scores.reshape(1, -1), k=10)
            all_ndcg.append(ndcg)
        return np.mean(all_ndcg)

    def _sweep(self):
        logger.info("sweeping hyperparams")
        np.random.seed(42)

        logger.info("sweeping stage 1")

        grid_group_1 = {
            "objective": ["rank:ndcg", "rank:map"],
            "learning_rate": [1e-2],
            "n_estimators": [400],
            "max_depth": [6, 8, 10, 12],
            "min_child_weight": [1, 2, 4],
            "subsample": [1],
            "colsample_bytree": [1],
            "tree_method": ["gpu_hist"],
            "random_state": [42],
            # "early_stopping_rounds": [20],  # removed because we don't really have a valid set on ranking
        }

        all_scores = []
        param_grid = ParameterGrid(grid_group_1)
        for p in tqdm(param_grid, total=len(param_grid)):
            ndcg = self.__train_and_eval(p)
            all_scores.append((p, ndcg))

        best_param, best_score = sorted(all_scores, key=lambda x: x[1], reverse=True)[0]

        logger.info("stage 1 best param and score")
        logger.info(best_param)
        logger.info(best_score)

        np.random.seed(42)
        logger.info("stage 2")

        grid_group_2 = {
            "objective": [best_param["objective"]],
            "learning_rate": [1e-2],
            "n_estimators": [400],
            "max_depth": [best_param["max_depth"]],
            "min_child_weight": [best_param["min_child_weight"]],
            "subsample": [1.0, 0.95, 0.9, 0.85],
            "colsample_bytree": [1.0, 0.95, 0.9, 0.85],
            "tree_method": ["gpu_hist"],
            "random_state": [42],
            # "early_stopping_rounds": [20],
        }

        all_scores = []
        param_grid = ParameterGrid(grid_group_2)
        for p in tqdm(param_grid, total=len(param_grid)):
            ndcg = self.__train_and_eval(p)
            all_scores.append((p, ndcg))

        best_param, best_score = sorted(all_scores, key=lambda x: x[1], reverse=True)[0]

        logger.info("stage 2 best param and score")
        logger.info(best_param)

        np.random.seed(42)
        logger.info("stage 3")

        grid_group_3 = {
            "objective": ["rank:ndcg"],
            "learning_rate": [1e-1, 5e-2, 1e-2, 5e-3, 1e-3],
            "n_estimators": [400, 800, 1000, 1200, 200],
            "max_depth": [best_param["max_depth"]],
            "min_child_weight": [best_param["min_child_weight"]],
            "subsample": [best_param["subsample"]],
            "colsample_bytree": [best_param["colsample_bytree"]],
            "tree_method": ["gpu_hist"],
            "random_state": [42],
            # "early_stopping_rounds": [20],
        }

        all_scores = []
        param_grid = ParameterGrid(grid_group_3)
        for p in tqdm(param_grid, total=len(param_grid)):
            ndcg = self.__train_and_eval(p)
            all_scores.append((p, ndcg))

        best_param, best_score = sorted(all_scores, key=lambda x: x[1], reverse=True)[0]
        logger.info("stage 3 best param and score")
        logger.info(best_param)
        return best_param

    def _sweep_and_train(self):
        best_param = self._sweep()
        model = self._train(**best_param)
        return model, best_param

    def batch_score(
        self,
        resume_repersentations: List[np.ndarray],
        job_representations: List[np.ndarray],
    ) -> np.ndarray:
        if self.merge_op == "concat":
            x = np.concatenate([resume_repersentations, job_representations], axis=1)
        else:
            raise NotImplementedError
        pred_probs = self.model.predict(x)
        return pred_probs

    def score(
        self, resume_repersentation: np.ndarray, job_representation: np.ndarray
    ) -> float:
        prob = self.batch_score([resume_repersentation], [job_representation])[0]
        return float(prob)


class DotProductMetric(Metric):
    def __init__(self, normalize=False):
        super().__init__()
        self.normalize = normalize  # true if we are computing accurarcy, false if we are computing ranking
        return

    def score(self, resume_repersentation: np.ndarray, job_representation: np.ndarray):
        if self.normalize:
            resume_repersentation = resume_repersentation / np.linalg.norm(
                resume_repersentation
            )
            job_representation = job_representation / np.linalg.norm(
                job_representation
            )
            # and rescale to [0, 1]
            score = np.dot(resume_repersentation, job_representation)
            score = (score + 1) / 2
        else:
            score = np.dot(resume_repersentation, job_representation)
        
        if np.isnan(score):
            # has nan, e.g. due to openai emebdding returning nan
            # replace nan with 0
            score = np.nan_to_num(score)
            print("nan detected in score, replaced with nan_to_num")
        return score

    def batch_score(self, resume_repersentations: List[np.ndarray], job_representations: List[np.ndarray]) -> np.ndarray:
        resume_repersentations_array = np.array(resume_repersentations)
        job_representations_array = np.array(job_representations)
        if self.normalize:
            resume_repersentations_array = resume_repersentations_array / np.linalg.norm(
                resume_repersentations_array, axis=1, keepdims=True
            )
            job_representations_array = job_representations_array / np.linalg.norm(
                job_representations_array, axis=1, keepdims=True
            )
            # and rescale to [0, 1]
            scores = np.sum(resume_repersentations_array * job_representations_array, axis=1)
            scores = (scores + 1) / 2
        else:
            scores = np.sum(resume_repersentations_array * job_representations_array, axis=1)
        
        if np.isnan(np.sum(scores)):
            # has nan, e.g. due to openai emebdding returning nan
            # replace nan with 0
            scores = np.nan_to_num(scores)
            print("nan detected in scores, replaced with nan_to_num")
        return scores


class PrecomputedMetric(Metric):
    def __init__(self, precomputed_scores: Dict[Tuple, float]) -> None:
        super().__init__()
        self.precomputed_scores = precomputed_scores
        return

    def score(self, resume_repersentation: str, job_representation: str) -> float:
        """takes in a resume_id under the disguise of resume_repersentation, and a job_id under the disguise of job_representation

        Args:
            resume_repersentation (str): resume_id
            job_representation (str): job_id

        Returns:
            float: precomputed score from the precomputed_scores dict using the key (resume_repersentation, job_representation)
        """
        key = (resume_repersentation, job_representation)
        return self.precomputed_scores[key]

    def batch_score(
        self,
        resume_repersentations: List[str],
        job_representations: List[str],
    ) -> np.ndarray:
        scores = []
        for resume_id, job_id in zip(resume_repersentations, job_representations):
            scores.append(self.score(resume_id, job_id))
        return np.array(scores)