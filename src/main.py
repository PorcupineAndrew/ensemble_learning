#!/usr/bin/env python3
#-*- coding: utf-8 -*-
# **********************************************************************
# * Description   : main process
# * Last change   : 10:19:54 2020-05-18
# * Author        : Yihao Chen
# * Email         : chenyiha17@mails.tsinghua.edu.cn
# * License       : www.opensource.org/licenses/bsd-license.php
# **********************************************************************

import os
import click
import numpy as np
import pandas as pd
from pathlib import Path
from data_load import data_load

SCRIPT_PATH = Path(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = SCRIPT_PATH.parent / "data"

@click.command()
@click.option("--word-dims", "-d", default=50, help="dims for word2vec embedding")
@click.option("--clf-type", "-t", default="dt", help="classifier type")
@click.option("--n-clf", "-c", default=50, help="number of classifiers for bagging")
@click.option("--n-iter", "-i", default=50, help="number of iterations for adaboost")
@click.option("--n-workers", "-n", default=6, help="number of parallel workers")
@click.option("--framework", "-f", default="adaboost", help="framework in enssemble training")
@click.option("--output-path", "-o", default="./result.csv", help="output path")
def main(word_dims, clf_type, n_workers, framework, **kwargs):
    df_train = data_load(
        DATA_PATH / "train.csv",
        word_dims, sep="\t", engine="c",
        w2v_model_path=DATA_PATH / "word2vec.model",
        save_path=DATA_PATH / "train_vec.csv")

    if framework == "adaboost":
        from adaboost import train_models
        pred = train_models(kwargs["n_iter"], df_train, clf_type)
    elif framework == "bagging":
        from bagging import train_models
        pred = train_models(df_train, clf_type, kwargs["n_clf"], n_workers)
    else: raise RuntimeError("bad framework")

    p = pred(df_train.drop("label", axis=1), n_workers)
    print(f"train dataset RMSE: {RMSE(p, df_train.label)}")
    del df_train, p

    df_test = data_load(
        DATA_PATH / "test.csv",
        word_dims, sep="\t", engine="c",
        w2v_model_path=DATA_PATH / "word2vec.model",
        save_path=DATA_PATH / "test_vec.csv")

    p = pred(df_test, n_workers)

    pd.DataFrame.from_dict({"id": np.arange(len(p))+1, "predicted": p}) \
        .to_csv(kwargs["output_path"], index=False)

def RMSE(pred, label):
    return np.mean((pred - label) ** 2) ** 0.5

if __name__ == "__main__":
    main()
