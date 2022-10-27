#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 21/03/22

@author: Nicolò Felicioni
"""
from inspect import isclass

import numpy as np

from dataclasses import dataclass
from typing import Optional, Tuple, Union

from obp.ope import RegressionModel, OffPolicyEvaluation
from obp.types import BanditFeedback
from pyieoe.evaluator import InterpretableOPEEvaluator
from pyieoe.utils import _choose_uniform
from sklearn.base import BaseEstimator, clone
from sklearn.model_selection._search import BaseSearchCV
from tqdm import tqdm
import json

from DeficientOffPolicyEvaluation import DeficientOffPolicyEvaluation


@dataclass
class DeficientSupportInterpretableOPEEvaluator(InterpretableOPEEvaluator):

    n_deficient_actions: int = 0
    is_cross_validation: bool = False

    def _choose_evaluation_policy(
            self,
            s: int,
    ) -> Tuple[float, np.ndarray]:
        """Randomly choose evaluation policy and resample using bootstrap."""
        np.random.seed(seed=s)
        idx = np.random.choice(len(self.evaluation_policies))
        ground_truth, action_dist = self.evaluation_policies[idx]
        action_dist = action_dist[self.bootstrap_idx]
        return ground_truth, action_dist



    def _supported_action_selection(self, bootstrap_bandit_feedback, bootstrap_action_dist, len_list, s):

        n_actions = bootstrap_bandit_feedback["n_actions"]


        n_supported_actions = n_actions - self.n_deficient_actions

        random_state = np.random.RandomState(seed=s)

        unique_contexts, context_ids_round = np.unique(
            bootstrap_bandit_feedback["context"], axis=0, return_inverse=True)

        n_unique_contexts = unique_contexts.shape[0]

        supported_actions = np.argsort(
            random_state.gumbel(size=(n_unique_contexts, n_actions)), axis=1
        )[:, ::-1][:, :n_supported_actions]

        supported_actions_round = supported_actions[context_ids_round]

        if n_supported_actions > 0:
            uniform_pscore_value = 1/n_supported_actions
        else:
            uniform_pscore_value = 0

        previous_n_rounds = bootstrap_bandit_feedback["n_rounds"]

        pscore = np.empty(shape=previous_n_rounds)

        pi_b_temp = np.zeros(shape=(previous_n_rounds, n_actions))

        np.put_along_axis(pi_b_temp, supported_actions_round, uniform_pscore_value, axis=1)

        pi_b = np.empty(shape=(previous_n_rounds, n_actions, len_list))

        for i in range(len_list):
            pi_b[:, :, i] = pi_b_temp

        for i, action in enumerate(bootstrap_bandit_feedback["action"]):
            if action not in supported_actions_round[i]:
                pscore[i] = 0
            else:
                pscore[i] = uniform_pscore_value


        round_mask = np.in1d(pscore, 0)
        round_mask = ~round_mask

        for key_ in self.bandit_feedback.keys():
            # if the size of a certain key_ is not equal to n_rounds,
            # we should not resample that certain key_
            # e.g. we want to resample action and reward, but not n_rounds
            if (
                    not isinstance(self.bandit_feedback[key_], np.ndarray)
                    or len(self.bandit_feedback[key_]) != self.bandit_feedback["n_rounds"]
            ):
                continue

            bootstrap_bandit_feedback[key_] = bootstrap_bandit_feedback[key_][
                round_mask
            ]


        bootstrap_bandit_feedback["n_rounds"] = round_mask.sum()
        bootstrap_bandit_feedback["pi_b"] = pi_b[round_mask]
        bootstrap_bandit_feedback["supported_actions_round"] = supported_actions_round[round_mask]

        bootstrap_action_dist = bootstrap_action_dist[round_mask]

        return bootstrap_bandit_feedback, bootstrap_action_dist, supported_actions, supported_actions_round[round_mask]



    def _sample_bootstrap_bandit_feedback_and_evaluation_policy(
                self, s: int, sample_size: Optional[int]
        ) -> Tuple[BanditFeedback, float, np.ndarray, np.ndarray, np.ndarray]:
            """Randomly sample bootstrap data from bandit_feedback."""
            bootstrap_bandit_feedback = self.bandit_feedback.copy()

            len_list = bootstrap_bandit_feedback["position"].max() + 1

            np.random.seed(seed=s)
            if sample_size is None:
                sample_size = self.bandit_feedback["n_rounds"]

            self.bootstrap_idx = np.random.choice(
                np.arange(sample_size), size=sample_size, replace=True
            )
            for key_ in self.bandit_feedback.keys():
                # if the size of a certain key_ is not equal to n_rounds,
                # we should not resample that certain key_
                # e.g. we want to resample action and reward, but not n_rounds
                if (
                        not isinstance(self.bandit_feedback[key_], np.ndarray)
                        or len(self.bandit_feedback[key_]) != self.bandit_feedback["n_rounds"]
                ):
                    continue
                bootstrap_bandit_feedback[key_] = bootstrap_bandit_feedback[key_][
                    self.bootstrap_idx
                ]

            bootstrap_bandit_feedback["n_rounds"] = sample_size

            idx_eval_policy = np.random.choice(len(self.evaluation_policies))
            ground_truth, action_dist = self.evaluation_policies[idx_eval_policy]
            bootstrap_action_dist = action_dist[self.bootstrap_idx]


            bootstrap_bandit_feedback, bootstrap_action_dist, supported_actions, supported_actions_round = self._supported_action_selection(
                                                                        bootstrap_bandit_feedback,
                                                                        bootstrap_action_dist,
                                                                        len_list,
                                                                        s)


            return bootstrap_bandit_feedback, ground_truth, bootstrap_action_dist, supported_actions, supported_actions_round


    def _estimate_policy_value_s(
            self,
            s: int,
            bootstrap_bandit_feedback: BanditFeedback,
            _regression_model: Union[BaseEstimator, BaseSearchCV],
            bootstrap_action_dist: np.ndarray,
            n_folds: int,
    ) -> Tuple[dict, np.ndarray]:
        """Estimates the policy values using selected ope estimators under a particular environments."""
        # prepare regression model for ope
        regression_model = RegressionModel(
            n_actions=self.bandit_feedback["n_actions"],
            len_list=int(self.bandit_feedback["position"].max() + 1),
            base_model=_regression_model,
            fitting_method="normal",
        )

        estimated_reward_by_reg_model = regression_model.fit_predict(
            context=bootstrap_bandit_feedback["context"],
            action=bootstrap_bandit_feedback["action"],
            reward=bootstrap_bandit_feedback["reward"],
            position=bootstrap_bandit_feedback["position"],
            pscore=bootstrap_bandit_feedback["pscore"],
            action_dist=bootstrap_action_dist,
            n_folds=n_folds,
            random_state=int(s),
        )

        # estimate policy value using ope
        ope = DeficientOffPolicyEvaluation(
            bandit_feedback=bootstrap_bandit_feedback,
            ope_estimators=self.ope_estimators,
        )
        estimated_policy_value = ope.estimate_policy_values(
            action_dist=bootstrap_action_dist,
            estimated_rewards_by_reg_model=estimated_reward_by_reg_model,
        )

        return estimated_policy_value, estimated_reward_by_reg_model



    def get_stats(
            self,
            action_context: np.ndarray,
            sample_size: Optional[int] = None,
    ) -> Tuple[dict, np.ndarray, np.ndarray]:

        self.policy_list = ['pi_e', 'pi_bar_cos', 'pi_bar_cluster',]
        self.renyi_dict = {
            policy: np.zeros(self.n_runs) for policy in self.policy_list
        }

        self.unsupported_features_array = np.zeros(self.n_runs)
        self.div_support_array = np.zeros(self.n_runs)

        n_actions, n_features = action_context.shape
        assert n_features == 40, print(f"n_features: {n_features}")

        for i, s in enumerate(tqdm(self.random_states)):
            np.random.seed(seed=s)
            # randomly select bandit_feedback
            self.bandit_feedback = self._choose_bandit_feedback(s)

            # randomly sample from selected bandit_feedback
            # randomly choose evaluation policy
            bootstrap_bandit_feedback, ground_truth, bootstrap_action_dist, supported_actions, supported_actions_round = \
                self._sample_bootstrap_bandit_feedback_and_evaluation_policy(
                    s, sample_size
                )

            action_dist_pos = bootstrap_action_dist[np.arange(bootstrap_action_dist.shape[0]),
                                                      :,
                                                    bootstrap_bandit_feedback['position']]

            div_support = (1 - np.take_along_axis(action_dist_pos, supported_actions_round, axis=1).sum(axis=1)).mean()

            self.div_support_array[i] = div_support


            n_unique_contexts = supported_actions.shape[0]
            assert n_actions >= supported_actions.shape[1], print(f"n_actions: {n_actions}, "
                                                                  f"supp_act_shape: {supported_actions.shape[1]}")


            supported_features = np.zeros((n_unique_contexts, n_features))
            for context in range(n_unique_contexts):
                curr_supported_actions = supported_actions[context]
                for act in curr_supported_actions:
                    supported_features[context] += action_context[act]


            self.unsupported_features_array[i] = (supported_features==0).sum(axis=1).mean()

            # randomly choose hyperparameters of ope estimators
            self._choose_ope_estimator_hyperparam(s)

            # estimate policy value using ope
            ope = DeficientOffPolicyEvaluation(
                bandit_feedback=bootstrap_bandit_feedback,
                ope_estimators=self.ope_estimators,
            )


            # calculate renyi
            renyi_s = ope.estimate_renyi_dict(action_dist=bootstrap_action_dist,)


            # store renyi results
            for policy in ['pi_e', 'pi_bar_cos', 'pi_bar_cluster',]:
                self.renyi_dict[policy][i] = renyi_s[policy]

                temp_res_dict = {policy: list(self.renyi_dict[policy])}
                with open(f"TEMP_{policy}_renyi_dict.json", "w") as fp:
                    json.dump(temp_res_dict, fp)


        return self.renyi_dict, self.unsupported_features_array, self.div_support_array



    def estimate_policy_value(
            self,
            n_folds_: Union[int, Optional[dict]] = 2,
            sample_size: Optional[int] = None,
    ) -> dict:
        """Estimates the policy values using selected ope estimators under a range of environments."""
        # initialize dictionaries to store results
        self.policy_value = {est: np.zeros(self.n_runs) for est in self.estimator_names}
        self.squared_error = {
            est: np.zeros(self.n_runs) for est in self.estimator_names
        }
        self.reg_model_metrics = {
            metric: np.zeros(self.n_runs) for metric in self.reg_model_metric_names
        }
        for i, s in enumerate(tqdm(self.random_states)):
            np.random.seed(seed=s)
            # randomly select bandit_feedback
            self.bandit_feedback = self._choose_bandit_feedback(s)

            if self.pscore_estimators is not None:
                # randomly choose pscore estimator
                pscore_estimator = np.random.choice(self.pscore_estimators)
                # randomly choose hyperparameters of pscore estimator
                if isinstance(pscore_estimator, BaseEstimator):
                    classifier = pscore_estimator
                    setattr(classifier, "random_state", s)
                elif isclass(pscore_estimator) and issubclass(
                        pscore_estimator, BaseEstimator
                ):
                    pscore_estimator_hyperparam = (
                        self._choose_pscore_estimator_hyperparam(s, pscore_estimator)
                    )
                    classifier = clone(pscore_estimator(**pscore_estimator_hyperparam))
                else:
                    raise ValueError(
                        f"pscore_estimator must be BaseEstimator or BaseSearchCV, but {type(pscore_estimator)} is given."
                    )
                # fit classifier
                classifier.fit(
                    self.bandit_feedback["context"], self.bandit_feedback["action"]
                )
                estimated_pscore = classifier.predict_proba(
                    self.bandit_feedback["context"]
                )
                # replace pscore in bootstrap bandit feedback with estimated pscore
                self.bandit_feedback["pscore"] = estimated_pscore[
                    np.arange(self.bandit_feedback["n_rounds"]),
                    self.bandit_feedback["action"],
                ]

            # randomly sample from selected bandit_feedback
            # randomly choose evaluation policy
            bootstrap_bandit_feedback, ground_truth, bootstrap_action_dist, _, _ =\
                self._sample_bootstrap_bandit_feedback_and_evaluation_policy(
                s, sample_size
            )


            # randomly choose hyperparameters of ope estimators
            self._choose_ope_estimator_hyperparam(s)

            # randomly choose regression model
            regression_model = self._choose_regression_model(s)
            # randomly choose hyperparameters of regression models
            if isinstance(regression_model, BaseEstimator):
                setattr(regression_model, "random_state", s)
            elif isclass(regression_model) and issubclass(
                    regression_model, BaseEstimator
            ):
                regression_model_hyperparam = self._choose_regression_model_hyperparam(
                    s, regression_model
                )
                regression_model = regression_model(**regression_model_hyperparam)
            else:
                raise ValueError(
                    f"regression_model must be BaseEstimator or BaseSearchCV, but {type(regression_model)} is given."
                )

            # randomly choose number of folds
            if isinstance(n_folds_, dict):
                n_folds = _choose_uniform(
                    s,
                    n_folds_["lower"],
                    n_folds_["upper"],
                    n_folds_["type"],
                )
            else:
                n_folds = n_folds_
            # estimate policy value using each ope estimator under setting s
            (
                policy_value_s,
                estimated_rewards_by_reg_model_s,
            ) = self._estimate_policy_value_s(
                s,
                bootstrap_bandit_feedback,
                regression_model,
                bootstrap_action_dist,
                n_folds,
            )
            # calculate squared error for each ope estimator
            squared_error_s = self._calculate_squared_error_s(
                policy_value_s,
                ground_truth,
            )
            # evaluate the performance of reg_model
            r_pred = estimated_rewards_by_reg_model_s[
                np.arange(bootstrap_bandit_feedback["n_rounds"]),
                bootstrap_bandit_feedback["action"],
                bootstrap_bandit_feedback["position"],
            ]

            # if there is more than one reward in the bandit feedback:
            if np.unique(bootstrap_bandit_feedback["reward"]).shape[0] > 1:
                reg_model_metrics = self._calculate_rec_model_performance_s(
                    r_true=bootstrap_bandit_feedback["reward"],
                    r_pred=r_pred,
                )
            else:
                print("Warning: only one value observed in the reward vector.")
                reg_model_metrics = np.zeros(shape=len(self.reg_model_metric_names))


            # store results
            for est in self.estimator_names:
                self.policy_value[est][i] = policy_value_s[est]
                self.squared_error[est][i] = squared_error_s[est]
                if not self.is_cross_validation:
                    temp_res_dict = {est: list(self.squared_error[est])}
                    import json
                    with open(f"TEMP_{est}_res_dict.json", "w") as fp:
                        json.dump(temp_res_dict, fp)

            for j, metric in enumerate(self.reg_model_metric_names):
                self.reg_model_metrics[metric][i] = reg_model_metrics[j].mean()

        return self.policy_value