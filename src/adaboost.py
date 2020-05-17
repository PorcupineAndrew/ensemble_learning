#!/usr/bin/env python3
#-*- coding: utf-8 -*-
# **********************************************************************
# * Description   : Ada boost M1
# * Last change   : 16:07:27 2020-05-17
# * Author        : Yihao Chen
# * Email         : chenyiha17@mails.tsinghua.edu.cn
# * License       : www.opensource.org/licenses/bsd-license.php
# **********************************************************************

import numpy as np
from joblib import Parallel, delayed
from classifier import *

def train_models(n_iter, df, clf_type):
    size = df.shape[0]
    weight = np.ones(size) / size
    models, betas = [], []
    train_func = dict(svm=SVM_train, dt=DT_train)[clf_type]
    pred_func = dict(svm=SVM_pred, dt=DT_pred)[clf_type]
    vec = df.drop("label", axis="columns")
    label = df.label
    for _ in range(n_iter):
        m = train_func(vec, label, sample_weight=weight)
        mis = pred_func(m, vec) != label
        err = weight[mis].sum()
        print(f"iter {_} err {err}")
        if err > 0.5:
            break
        beta = err / (1-err)
        betas.append(beta)
        weight[~mis] *= beta
        weight /= weight.sum()
        models.append(m)
    voting_weight = np.log(1/np.array(betas))
    return lambda _df, _n_workers: pred(_df, models, voting_weight, clf_type, _n_workers)

def pred(df, clfs, voting_weight, clf_type, n_workers):
    options = {
        "n_jobs": n_workers,
        "backend": "multiprocessing",
        "verbose": 100,
    }
    preds = Parallel(**options)(delayed(dict(svm=SVM_pred, dt=DT_pred)[clf_type])(
                    clf, df) for clf in clfs)
    def vote(x):
        candi = np.unique(x)
        return candi[np.argmax([voting_weight[x == i].sum() for i in candi])]

    preds = np.apply_along_axis(vote, axis=0, arr=preds)
    return preds
