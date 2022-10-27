#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 12/01/22

@author: Nicolò Felicioni
"""
from dataclasses import dataclass
from functools import partial
from typing import Optional, Tuple

import numpy as np
from obp.ope import BaseOffPolicyEstimator
from sklearn.metrics import jaccard_score
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial import distance

@dataclass
class DiceSelfNormPureSimilarityIPS:

    action_context: np.ndarray
    estimator_name: str = "sn_pure_ips"

    def __post_init__(self):
        self.similarity = self._get_similarity(self.action_context)


    def estimate_policy_value(
            self,
            supported_actions_round: np.ndarray,
            reward: np.ndarray,
            action: np.ndarray,
            pscore: np.ndarray,
            action_dist: np.ndarray,
            position: Optional[np.ndarray] = None,
            **kwargs,
    ) -> float:

        round_results, round_norm = self._estimate_round_rewards(supported_actions_round=supported_actions_round,
                                                                 reward=reward,
                                                                 action=action,
                                                                 position=position,
                                                                 pscore=pscore,
                                                                 action_dist=action_dist,)
        return round_results.sum() / round_norm.sum()


    def _estimate_round_rewards(
            self,
            supported_actions_round: np.ndarray,
            reward: np.ndarray,
            action: np.ndarray,
            pscore: np.ndarray,
            action_dist: np.ndarray,
            position: Optional[np.ndarray] = None,
            **kwargs,
    ) ->  Tuple[np.ndarray, np.ndarray]:

        n_rounds = action_dist.shape[0]

        action_dist_pos = action_dist[np.arange(n_rounds), :, position]

        n_supported_actions = supported_actions_round.shape[1]
        pi_bar = np.empty(n_rounds)
        for i in np.arange(n_rounds):
            curr_action = action[i]
            curr_similarity_vector = self.similarity[:, curr_action]
            curr_supported_actions = supported_actions_round[i]
            normalization_vector = self.similarity[:, curr_supported_actions].sum(axis=1)

            curr_similarity_vector = np.divide(curr_similarity_vector, normalization_vector,
                                               out=np.array([1/n_supported_actions]*curr_similarity_vector.shape[0]),
                                               where=(normalization_vector!=0))

            pi_bar[i] = action_dist_pos[i].dot(curr_similarity_vector)


        return pi_bar * reward / pscore, pi_bar / pscore


    def _get_similarity(self, action_context) -> np.ndarray:

        # action_context shape: n_actions x n_features

        similarity = np.empty(shape=(action_context.shape[0], action_context.shape[0]))

        # TODO TO BE OPTIMIZED
        for i in range(action_context.shape[0]):
            for j in range(i, action_context.shape[0]):
                score = distance.dice(action_context[i], action_context[j])
                similarity[i, j] = score
                if i != j:
                    similarity[j, i] = score

        return similarity