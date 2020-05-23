#!/usr/bin/env python3
#-*- coding: utf-8 -*-

import os
import pandas as pd
import numpy as np
import click
from text_processing_util import TextProcessing
from text_cnn import kimCNN
from pathlib import Path
from keras.callbacks import ModelCheckpoint
from keras.models import load_model

MAX_SEQUENCE_LENGTH = 50
MAX_NB_WORDS = 10000
EMBEDDING_DIM = 300
VALIDATION_SPLIT = 0.2

SCRIPT_PATH = Path(os.path.abspath(__file__)).parent.parent
DATA_PATH = SCRIPT_PATH.parent / "data"
TRAIN_PATH = SCRIPT_PATH.parent / "train"

@click.command()
@click.option("--prefix", "-p", default="", help="prefix for experiment")
@click.option("--is-pred", is_flag=True, help="train or inference")
@click.option("--model-name", "-l", default=None, help="model name to load")
@click.option("--weight", "-w", default=None, help="weight to load")
@click.option("--output-path", "-o", default="pred.csv", help="path to predict output")
def main(prefix, model_name, weight, output_path, **kwargs):
    if not kwargs.pop("is_pred", False):
        df = pd.read_csv(DATA_PATH / "train.csv", sep="\t", engine="c")
        labels = df.overall.values.astype(int)
        # texts = df.summary.values.astype(str)
        texts = df.reviewText.values.astype(str)
        del df

        tp = TextProcessing(texts, labels, EMBEDDING_DIM, MAX_SEQUENCE_LENGTH, MAX_NB_WORDS, VALIDATION_SPLIT)

        x_train, y_train, x_val, y_val, word_index = tp.preprocess()
        with open(TRAIN_PATH / f"{prefix}tokenizer.json", "w") as f:
            f.write(tp.tokenizer_string)
        embeddings_index = tp.build_embedding_index_from_word2vec(DATA_PATH / "word2vec.model")

        labels_index = tp.labels_index
        del tp

        if model_name is None:
            model = kimCNN(EMBEDDING_DIM, MAX_SEQUENCE_LENGTH, MAX_NB_WORDS, embeddings_index, word_index, labels_index=labels_index)
        else:
            model = load_model(str(TRAIN_PATH / model_name))
        print(model.summary())

        # checkpoint 
        filepath = str(TRAIN_PATH / f"{prefix}weights.best.hdf5")
        checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')

        model.fit(x=x_train, y=y_train, batch_size=50, epochs=100, validation_data=(x_val, y_val), callbacks=[checkpoint])
        model.save(str(TRAIN_PATH / f"{prefix}model.h5"))
    else:
        model = load_model(str(TRAIN_PATH / model_name))
        print(model.summary())
        
        if weight is not None:
            model.load_weights(str(TRAIN_PATH / weight))

        df = pd.read_csv(DATA_PATH / "test.csv", sep="\t", engine="c")
        # texts = df.summary.values.astype(str)
        texts = df.reviewText.values.astype(str)
        labels = np.zeros(len(texts))
        del df

        with open(TRAIN_PATH / f"{prefix}tokenizer.json", "r") as f:
            tokenizer_string = f.read()
        tp = TextProcessing(texts, labels, EMBEDDING_DIM, MAX_SEQUENCE_LENGTH, MAX_NB_WORDS)
        x_test, _, _, _, word_index = tp.preprocess(tokenizer_string)

        pred = np.argmax(model.predict(x_test), 1) + 1
        pd.DataFrame.from_dict({"id": np.arange(len(pred))+1, "predicted": pred}) \
            .to_csv(output_path, index=False)

if __name__ == "__main__":
    main()
