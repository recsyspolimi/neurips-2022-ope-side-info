#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 09/01/22

@author: Nicolò Felicioni
"""

from typing import Optional, Tuple
from jax import vmap

import jax.numpy as jnp
import numpy as np
from jax import jit


def _convert_to_jnp(tuple_of_arrays: Tuple):
    list_of_jnp_arrays = []
    for arr in tuple_of_arrays:
        list_of_jnp_arrays.append(jnp.array(arr))

    return tuple(list_of_jnp_arrays)


def get_gamma_pinv(pi_b_pos, feature_tensor):

    gamma = jnp.tensordot(pi_b_pos, feature_tensor, axes=(1, 0))
    gamma_pinv = jnp.linalg.pinv(gamma[:])
    return gamma_pinv

get_gamma_pinv_compiled = jit(get_gamma_pinv)

class SelfNormSideInformationPseudoInverse:

    estimator_name: str = "snsipi"


    def init_feature_outer_product_tensor(self,
                                          action_context: np.ndarray):
        action_context = jnp.array(action_context)
        f = vmap(lambda x: jnp.outer(x, x))
        feature_tensor = f(action_context)

        return feature_tensor


    def get_q(self, action_dist_pos, action_context):
        return action_dist_pos.dot(action_context)

    def get_action_context_round(self, action: jnp.ndarray, action_context: jnp.ndarray):

        f = vmap(lambda a: action_context[a])
        action_context_round = f(action)

        return action_context_round



    def estimate_policy_value(
            self,
            reward: np.ndarray,
            action: np.ndarray,
            pscore: np.ndarray,
            pi_b: np.ndarray,
            action_dist: np.ndarray,
            action_context: np.ndarray,
            position: Optional[np.ndarray] = None,
            **kwargs,
    ) -> float:

        reward,\
        action,\
        pi_b,\
        action_dist,\
        action_context,\
        position = _convert_to_jnp((reward,action,pi_b,action_dist,action_context,position))

        n_rounds = action_dist.shape[0]
        action_dist_pos = action_dist[jnp.arange(n_rounds), :, position]

        if len(pi_b.shape) == 2:
            pi_b_pos = pi_b[jnp.arange(n_rounds), :]
        elif len(pi_b.shape) == 3:
            pi_b_pos = pi_b[jnp.arange(n_rounds), :, position]
        else:
            raise ValueError(f"pi_b has shape: {pi_b.shape}")

        feature_tensor = self.init_feature_outer_product_tensor(action_context)

        # gamma_pinv = get_gamma_pinv(pi_b_pos, feature_tensor)
        gamma_pinv = get_gamma_pinv_compiled(pi_b_pos, feature_tensor)
        q = self.get_q(action_dist_pos, action_context)
        action_context_round = self.get_action_context_round(action, action_context)

        temp_tensor = jnp.matmul(gamma_pinv, q.reshape((q.shape[0], q.shape[1], 1)))

        pi_weight = jnp.matmul(
            action_context_round.reshape((action_context_round.shape[0],
                                          1,
                                          action_context_round.shape[1])),
            temp_tensor)

        pi_weight = pi_weight.reshape(reward.shape[0])

        scores = (pi_weight * reward)

        return np.float(scores.sum() / pi_weight.sum())



