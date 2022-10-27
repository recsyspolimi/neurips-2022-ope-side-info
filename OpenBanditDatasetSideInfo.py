#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 09/01/22

@author: Nicolò Felicioni
"""
from dataclasses import dataclass

from obp.dataset import OpenBanditDataset
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np

@dataclass
class OpenBanditDatasetSideInfo(OpenBanditDataset):

    def pre_process(self) -> None:
        """Preprocess raw open bandit dataset.

        Note
        -----
        This is the default feature engineering and please override this method to
        implement your own preprocessing.
        see https://github.com/st-tech/zr-obp/blob/master/examples/examples_with_obd/custom_dataset.py for example.

        """
        user_cols = self.data.columns.str.contains("user_feature")
        self.context = pd.get_dummies(
            self.data.loc[:, user_cols], drop_first=True
        ).values
        item_feature_0 = self.item_context["item_feature_0"]
        # item_feature_cat = self.item_context.drop("item_feature_0", 1).apply(
        #     LabelEncoder().fit_transform
        # )
        item_feature_cat = pd.get_dummies(self.item_context[['item_feature_1', 'item_feature_2', 'item_feature_3']])

        self.action_context = pd.concat([item_feature_cat, item_feature_0], axis=1).values

        self.unique_context = np.unique(self.context, axis=0)
