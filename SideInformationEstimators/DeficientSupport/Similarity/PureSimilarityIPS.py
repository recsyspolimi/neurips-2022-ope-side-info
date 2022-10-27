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
from sklearn.metrics.pairwise import cosine_similarity



@dataclass
class PureSimilarityIPS:
    action_context: np.ndarray
    estimator_name: str = "pure_ips"

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

        round_results = self._estimate_round_rewards(supported_actions_round=supported_actions_round,
                                                     reward=reward,
                                                     action=action,
                                                     position=position,
                                                     pscore=pscore,
                                                     action_dist=action_dist,)
        return round_results.mean()


    def _estimate_round_rewards(
            self,
            supported_actions_round: np.ndarray,
            reward: np.ndarray,
            action: np.ndarray,
            pscore: np.ndarray,
            action_dist: np.ndarray,
            position: Optional[np.ndarray] = None,
            **kwargs,
    ) -> np.ndarray:

        n_rounds = action_dist.shape[0]

        action_dist_pos = action_dist[np.arange(n_rounds), :, position]

        pi_bar = np.empty(shape=n_rounds)
        n_supported_actions = supported_actions_round.shape[1]
        for i in np.arange(n_rounds):
            curr_action = action[i]
            curr_similarity_vector = self.similarity[:, curr_action]
            curr_supported_actions = supported_actions_round[i]
            normalization_vector = self.similarity[:, curr_supported_actions].sum(axis=1)

            curr_similarity_vector = np.divide(curr_similarity_vector, normalization_vector,
                                               out=np.array([1/n_supported_actions]*curr_similarity_vector.shape[0]),
                                               where=(normalization_vector!=0))

            pi_bar[i] = action_dist_pos[i].dot(curr_similarity_vector)

        return pi_bar * reward / pscore



    def _get_similarity(self, action_context) -> np.ndarray:
        # return cosine_similarity(action_context)

        similarity = cosine_similarity(action_context)
        similarity = (similarity.T / similarity.sum(axis=1)).T
        return similarity


@dataclass
class SelfNormPureSimilarityIPS:

    action_context: np.ndarray
    estimator_name: str = "sn_pure_ips_2"

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

        if position is None:
            position = np.zeros(shape=n_rounds, dtype=np.int8)

        action_dist_pos = action_dist[np.arange(n_rounds), :, position]

        pi_bar = np.empty(shape=n_rounds)
        n_supported_actions = supported_actions_round.shape[1]

        for i in np.arange(n_rounds):
            curr_action = action[i]
            curr_similarity_vector = self.similarity[:, curr_action]
            curr_supported_actions = supported_actions_round[i]
            normalization_vector = self.similarity[:, curr_supported_actions].sum(axis=1)

            curr_similarity_vector = np.divide(curr_similarity_vector, normalization_vector,
                                               out=np.array([1/n_supported_actions]*curr_similarity_vector.shape[0]),
                                               where=(normalization_vector!=0))

            pi_bar[i] = action_dist_pos[i].dot(curr_similarity_vector)


        self.pi_bar = pi_bar

        return pi_bar * reward / pscore, pi_bar / pscore


    def _get_similarity(self, action_context) -> np.ndarray:
        # return cosine_similarity(action_context)

        similarity = cosine_similarity(action_context)
        similarity = (similarity.T / similarity.sum(axis=1)).T
        return similarity