#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 11/01/22

@author: Nicolò Felicioni
"""

from typing import Dict, Tuple
from typing import Optional

from sklearn.cluster import KMeans

import numpy as np

class InverseProbabilityWeightingClustering():


    estimator_name: str = "ips_cluster"

    def __init__(self, n_clusters: int, action_context: np.ndarray,):

        # normalization of action context in order to have euclidean
        # distance linearly connected with cosine similarity
        # ref: https://stats.stackexchange.com/questions/299013/cosine-distance-as-similarity-measure-in-kmeans

        # norm_arr shape : |number of actions|
        norm_arr = 1 / ((action_context ** 2).sum(axis=1))

        self.n_clusters = n_clusters
        self.n_actions = action_context.shape[0]
        self.normalized_action_context = (action_context.T * norm_arr).T
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(self.normalized_action_context)
        self.kmeans: KMeans
        self.action_to_cluster = self.kmeans.predict(self.normalized_action_context)



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

        self.kmeans: KMeans
        action_cluster = self.kmeans.predict(self.normalized_action_context[action])

        action_dist_cluster = np.empty(shape=(action_dist.shape[0], self.n_clusters, action_dist.shape[2]))

        for cluster in range(self.n_clusters):
            action_dist_cluster[:, cluster, :] = action_dist[:, self.action_to_cluster == cluster, :].sum(axis=1)

        n_rounds = action_dist.shape[0]

        supported_clusters_round = self.action_to_cluster[supported_actions_round[np.arange(n_rounds)]]

        supported_clusters_mask_round = np.empty(shape=supported_clusters_round.shape)

        # this may be improved by vectorization
        for i in np.arange(n_rounds):
            supported_clusters_mask_round[i] = supported_clusters_round[i] == action_cluster[i]

        supported_clusters_number_round = supported_clusters_mask_round.sum(axis=1)

        if position is None:
            position = np.zeros(action_dist.shape[0], dtype=int)

        iw = action_dist_cluster[np.arange(action_cluster.shape[0]), action_cluster, position] / (pscore * supported_clusters_number_round)

        return reward * iw

    def estimate_policy_value(
            self,
            supported_actions_round: np.ndarray,
            reward: np.ndarray,
            action: np.ndarray,
            pscore: np.ndarray,
            action_dist: np.ndarray,
            position: Optional[np.ndarray] = None,
            **kwargs,
    ) -> np.ndarray:

        if position is None:
            position = np.zeros(action_dist.shape[0], dtype=int)

        return self._estimate_round_rewards(supported_actions_round=supported_actions_round,
                                            reward=reward, action=action, pscore=pscore, action_dist=action_dist,
                                            position=position).mean()


class SelfNormInverseProbabilityWeightingClustering():


    estimator_name: str = "snips_cluster"

    def __init__(self, n_clusters: int, action_context: np.ndarray):

        # normalization of action context in order to have euclidean
        # distance linearly connected with cosine similarity
        # ref: https://stats.stackexchange.com/questions/299013/cosine-distance-as-similarity-measure-in-kmeans

        # norm_arr shape : |number of actions|
        norm_arr = 1 / ((action_context ** 2).sum(axis=1))

        self.n_clusters = n_clusters
        self.n_actions = action_context.shape[0]
        self.normalized_action_context = (action_context.T * norm_arr).T
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(self.normalized_action_context)
        self.kmeans: KMeans
        self.action_to_cluster = self.kmeans.predict(self.normalized_action_context)



    def _estimate_round_rewards(
            self,
            supported_actions_round: np.ndarray,
            reward: np.ndarray,
            action: np.ndarray,
            pscore: np.ndarray,
            action_dist: np.ndarray,
            position: Optional[np.ndarray] = None,
            **kwargs,
    ) -> Tuple[np.ndarray, np.ndarray]:

        self.kmeans: KMeans
        action_cluster = self.kmeans.predict(self.normalized_action_context[action])

        action_dist_cluster = np.empty(shape=(action_dist.shape[0], self.n_clusters, action_dist.shape[2]))

        for cluster in range(self.n_clusters):
            action_dist_cluster[:, cluster, :] = action_dist[:, self.action_to_cluster == cluster, :].sum(axis=1)



        if position is None:
            position = np.zeros(action_dist.shape[0], dtype=int)

        n_rounds = action.shape[0]
        supported_clusters_round = self.action_to_cluster[supported_actions_round[np.arange(n_rounds)]]

        supported_clusters_mask_round = np.empty(shape=supported_clusters_round.shape)

        # this may be improved by vectorization
        for i in np.arange(n_rounds):
            supported_clusters_mask_round[i] = supported_clusters_round[i] == action_cluster[i]

        supported_clusters_number_round = supported_clusters_mask_round.sum(axis=1)

        self.pi_bar = action_dist_cluster[np.arange(action_cluster.shape[0]), action_cluster, position] / supported_clusters_number_round

        iw = action_dist_cluster[np.arange(action_cluster.shape[0]), action_cluster, position] / (pscore * supported_clusters_number_round)

        return reward * iw, iw

    def estimate_policy_value(
            self,
            supported_actions_round: np.ndarray,
            reward: np.ndarray,
            action: np.ndarray,
            pscore: np.ndarray,
            action_dist: np.ndarray,
            position: Optional[np.ndarray] = None,
            **kwargs,
    ) -> np.ndarray:

        if position is None:
            position = np.zeros(action_dist.shape[0], dtype=int)

        round_scores, round_norm = self._estimate_round_rewards(supported_actions_round=supported_actions_round,
            reward=reward, action=action, pscore=pscore, action_dist=action_dist,
                                                                position=position)
        return round_scores.sum() / round_norm.sum()
