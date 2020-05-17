#!/usr/bin/env python3
#-*- coding: utf-8 -*-
# **********************************************************************
# * Description   : load data and process
# * Last change   : 14:14:04 2020-05-11
# * Author        : Yihao Chen
# * Email         : chenyiha17@mails.tsinghua.edu.cn
# * License       : www.opensource.org/licenses/bsd-license.php
# **********************************************************************

import pandas as pd
import numpy as np
import os
from gensim.models import word2vec

def word2vec_model(words, size):
    config = {
        "sg": 0,
        "hs": 1,
        "min_count": 1,
        "window": 5,
        "size": size,
    }
    return word2vec.Word2Vec(words, **config)

def get_onehot_vec(words, size, model):
    zero_vec = np.zeros(size)
    vec = list(map(lambda w: model[w] if w in model else zero_vec, words))
    return np.mean(vec, axis=0)

def data_load(path, w2v_dims, save_path=None, **kwargs):
    if save_path is not None and os.path.exists(save_path):
        return pd.read_csv(save_path)

    df = pd.read_csv(path, **kwargs)

    for col in ["reviewerID", "asin", "unixReviewTime"]:
        _col = np.unique(df[col])
        _mapping = dict(zip(_col, np.arange(len(_col))))
        df[col] = list(map(lambda x: _mapping[x], df[col]))
        del _col, _mapping

    if "overall" in df.columns:
        df["label"] = df.overall
        df.drop("overall", axis="columns", inplace=True)

    if "id" in df.columns:
        df.drop("id", axis="columns", inplace=True)
    
    summary_words = [i.split(" ") for i in df.summary.astype(str)]
    print("word2vec: summary words...", end="")
    w2v_model = word2vec_model(summary_words, w2v_dims)
    df = df.join(pd.DataFrame(list(map(
                    lambda x: get_onehot_vec(x, w2v_dims, w2v_model), summary_words)),
                    columns=[f"_s{i}_" for i in range(w2v_dims)]))
    df.drop("summary", axis="columns", inplace=True)
    del summary_words, w2v_model
    print("finished")

    reviewText_words = [i.split(" ") for i in df.reviewText.astype(str)]
    w2v_model = word2vec_model(reviewText_words, w2v_dims)
    print("word2vec: reviewText words...", end="")
    df = df.join(pd.DataFrame(list(map(
                    lambda x: get_onehot_vec(x, w2v_dims, w2v_model), reviewText_words)),
                    columns=[f"_r{i}_" for i in range(w2v_dims)]))
    df.drop("reviewText", axis="columns", inplace=True)
    del reviewText_words, w2v_model
    print("finished")

    if save_path is not None:
        df.to_csv(save_path, index=False)

    return df

# path = "../data/train.csv"
# df = data_load(path, 50, sep="\t", engine="c", save_path="./train_vec.csv")
