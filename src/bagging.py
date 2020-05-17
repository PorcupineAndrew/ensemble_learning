#!/usr/bin/env python3
#-*- coding: utf-8 -*-
# **********************************************************************
# * Description   : bagging framework
# * Last change   : 16:53:19 2020-05-11
# * Author        : Yihao Chen
# * Email         : chenyiha17@mails.tsinghua.edu.cn
# * License       : www.opensource.org/licenses/bsd-license.php
# **********************************************************************

import numpy as np
import pandas as pd
import time
from data_load import data_load
from classifier import *
from joblib import Parallel, delayed

def one_bootstrap_executor(df, clf_type):
    size = df.shape[0]
    _df = df.iloc[np.random.randint(0, size, size)]
    return dict(svm=SVM_train, dt=DT_train)[clf_type](_df.drop("label", axis="columns"), _df.label)

def train_models(df, clf_type, n_model, n_workers):
    options = {
        "n_jobs": n_workers,
        "backend": "multiprocessing",
        "verbose": 100,
    }
    models = Parallel(**options)(delayed(one_bootstrap_executor)(df, clf_type)
                                                            for i in range(n_model))
    return lambda _df, _n_workers: pred(_df, models, clf_type, _n_workers)

def pred(df, clfs, clf_type, n_workers):
    options = {
        "n_jobs": n_workers,
        "backend": "multiprocessing",
        "verbose": 100,
    }
    preds = Parallel(**options)(delayed(dict(svm=SVM_pred, dt=DT_pred)[clf_type])(
                    clf, df) for clf in clfs)
    preds = np.array([np.bincount(i).argmax() for i in np.array(preds).T.tolist()])
    return preds

# df = data_load(None, 50, sep="\t", engine="c", save_path="./train_vec.csv")
# df.drop([f"_r{i}_" for i in range(50)], axis=1, inplace=True)
# clfs = train_models(df, "dt", 10, 6)
# df_test = df
# p = pred(df_test.drop("label", axis=1), clfs, "dt", 6)
# print(np.mean((p - df_test.label.values) ** 2) ** 0.5)
