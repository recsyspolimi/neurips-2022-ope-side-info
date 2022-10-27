#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 22/03/22

@author: Nicolò Felicioni
"""
from dataclasses import dataclass
from typing import Optional, Dict, Union

from obp.ope import OffPolicyEvaluation
import numpy as np
from obp.utils import check_array

from SideInformationEstimators.DeficientSupport.Clustering.IPSClustering import InverseProbabilityWeightingClustering, \
    SelfNormInverseProbabilityWeightingClustering
from SideInformationEstimators.DeficientSupport.PseudoInverse.SideInformationPseudoInverse import \
    SelfNormSideInformationPseudoInverse
from SideInformationEstimators.DeficientSupport.Similarity.DicePureSimilarityIPS import DiceSelfNormPureSimilarityIPS
from SideInformationEstimators.DeficientSupport.Similarity.JaccardPureSimilarityIPS import \
    JaccardSelfNormPureSimilarityIPS


from SideInformationEstimators.DeficientSupport.Similarity.PureSimilarityIPS import PureSimilarityIPS, \
    SelfNormPureSimilarityIPS


def _get_renyi(p: np.ndarray, q: np.ndarray):

    assert p.shape == q.shape, print(f"p shape: {p.shape}\n"
                                     f"q shape: {q.shape}")

    assert len(p.shape) == 1

    renyi_round = (p * p) / (q * q)

    return renyi_round.mean()



@dataclass
class DeficientOffPolicyEvaluation(OffPolicyEvaluation):

    def _create_estimator_inputs(
            self,
            action_dist: np.ndarray,
            estimated_rewards_by_reg_model: Optional[
                Union[np.ndarray, Dict[str, np.ndarray]]
            ] = None,
            estimated_pscore: Optional[Union[np.ndarray, Dict[str, np.ndarray]]] = None,
            estimated_importance_weights: Optional[
                Union[np.ndarray, Dict[str, np.ndarray]]
            ] = None,
    ) -> Dict[str, Dict[str, np.ndarray]]:
        """Create input dictionary to estimate policy value using subclasses of `BaseOffPolicyEstimator`"""
        check_array(array=action_dist, name="action_dist", expected_dim=3)
        if estimated_rewards_by_reg_model is None:
            pass
        elif isinstance(estimated_rewards_by_reg_model, dict):
            for estimator_name, value in estimated_rewards_by_reg_model.items():
                check_array(
                    array=value,
                    name=f"estimated_rewards_by_reg_model[{estimator_name}]",
                    expected_dim=3,
                )
                if value.shape != action_dist.shape:
                    raise ValueError(
                        f"Expected `estimated_rewards_by_reg_model[{estimator_name}].shape == action_dist.shape`, but found it False."
                    )
        elif estimated_rewards_by_reg_model.shape != action_dist.shape:
            raise ValueError(
                f"Expected `estimated_rewards_by_reg_model.shape == action_dist.shape`, but found it False\n"
                f"estimated_rewards_by_reg_model.shape: {estimated_rewards_by_reg_model.shape}\n"
                f"action_dist.shape: {action_dist.shape}"
            )
        for var_name, value_or_dict in {
            "estimated_pscore": estimated_pscore,
            "estimated_importance_weights": estimated_importance_weights,
        }.items():
            if value_or_dict is None:
                pass
            elif isinstance(value_or_dict, dict):
                for estimator_name, value in value_or_dict.items():
                    check_array(
                        array=value,
                        name=f"{var_name}[{estimator_name}]",
                        expected_dim=1,
                    )
                    if value.shape[0] != action_dist.shape[0]:
                        raise ValueError(
                            f"Expected `{var_name}[{estimator_name}].shape[0] == action_dist.shape[0]`, but found it False"
                        )
            else:
                check_array(array=value_or_dict, name=var_name, expected_dim=1)
                if value_or_dict.shape[0] != action_dist.shape[0]:
                    raise ValueError(
                        f"Expected `{var_name}.shape[0] == action_dist.shape[0]`, but found it False"
                    )

        estimator_inputs = {
            estimator_name: {
                input_: self.bandit_feedback[input_]
                for input_ in ["reward", "action", "position"]
            }
            for estimator_name in self.ope_estimators_
        }

        for estimator_name in self.ope_estimators_:
            if "pscore" in self.bandit_feedback:
                estimator_inputs[estimator_name]["pscore"] = self.bandit_feedback[
                    "pscore"
                ]
            else:
                estimator_inputs[estimator_name]["pscore"] = None

            estimator_inputs[estimator_name]["action_dist"] = action_dist
            estimator_inputs = self._preprocess_model_based_input(
                estimator_inputs=estimator_inputs,
                estimator_name=estimator_name,
                model_based_input={
                    "estimated_rewards_by_reg_model": estimated_rewards_by_reg_model,
                    "estimated_pscore": estimated_pscore,
                    "estimated_importance_weights": estimated_importance_weights,
                },
            )

            for estimator_name, estimator in self.ope_estimators_.items():
                if isinstance(estimator, (InverseProbabilityWeightingClustering,
                                          SelfNormInverseProbabilityWeightingClustering,
                                          PureSimilarityIPS, SelfNormPureSimilarityIPS,
                                          JaccardSelfNormPureSimilarityIPS,
                                          DiceSelfNormPureSimilarityIPS
                                          )):
                    estimator_inputs[estimator_name]["supported_actions_round"] =\
                        self.bandit_feedback["supported_actions_round"]
                elif isinstance(estimator, SelfNormInverseProbabilityWeightingClustering):
                    estimator_inputs[estimator_name]["pi_b"] = self.bandit_feedback["pi_b"]
                elif isinstance(estimator, SelfNormSideInformationPseudoInverse):
                    estimator_inputs[estimator_name]["pi_b"] = self.bandit_feedback["pi_b"]
                    estimator_inputs[estimator_name]["action_context"] = self.bandit_feedback["action_context"]


        return estimator_inputs



    def estimate_renyi_dict(
            self,
            action_dist: np.ndarray,
            estimated_rewards_by_reg_model: Optional[
                Union[np.ndarray, Dict[str, np.ndarray]]
            ] = None,
            estimated_pscore: Optional[Union[np.ndarray, Dict[str, np.ndarray]]] = None,
            estimated_importance_weights: Optional[
                Union[np.ndarray, Dict[str, np.ndarray]]
            ] = None,
    ) -> Dict[str, float]:

        renyi_dict = {"pi_e": None,
                     'pi_bar_cos': None,
                     'pi_bar_cluster': None}

        estimator_inputs = self._create_estimator_inputs(
            action_dist=action_dist,
            estimated_rewards_by_reg_model=estimated_rewards_by_reg_model,
            estimated_pscore=estimated_pscore,
            estimated_importance_weights=estimated_importance_weights,
        )

        n_rounds = self.bandit_feedback['reward'].shape[0]
        position = self.bandit_feedback['position']
        action = self.bandit_feedback['action']

        # pi_b_pos = self.bandit_feedback['pi_b'][np.arange(n_rounds), :, position]
        pscore = self.bandit_feedback['pscore']

        for estimator_name, estimator in self.ope_estimators_.items():
            estimator._estimate_round_rewards(**estimator_inputs[estimator_name])
            # COSINE
            if isinstance(estimator, SelfNormPureSimilarityIPS):
                renyi_dict['pi_bar_cos'] = _get_renyi(estimator.pi_bar, pscore)

            elif isinstance(estimator, SelfNormInverseProbabilityWeightingClustering):
                renyi_dict['pi_bar_cluster'] = _get_renyi(estimator.pi_bar, pscore)

            renyi_dict['pi_e'] = _get_renyi(action_dist[np.arange(n_rounds), action, position], pscore)

        return renyi_dict