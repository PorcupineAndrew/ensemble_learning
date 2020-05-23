#!/usr/bin/env python3
#-*- coding:utf-8 -*-
# **********************************************************************
# * Description   : classifier
# * Last change   : 16:32:22 2020-05-23
# * Author        : Yihao Chen
# * Email         : chenyiha17@mails.tsinghua.edu.cn
# * License       : www.opensource.org/licenses/bsd-license.php
# **********************************************************************

from sklearn import svm, tree

def SVM_train(train_data, train_labels, **kwargs):
    '''train SVM classifier

    Args:
    --------
    train_data: train data vector
    train_labels: train data labels

    Returns:
    --------
    clf: SVM classifer
    '''
    clf = svm.LinearSVC(dual=False, class_weight="balanced")
    clf = clf.fit(train_data, train_labels, **kwargs)
    return clf

def SVM_pred(clf, testing_data):
    '''prediction by SVM classifier

    Args:
    --------
    clf: SVM classifier
    testing_data: testing data vector

    Returns:
    --------
    prediction: prediction list
    '''
    prediction = list(clf.predict(testing_data))
    assert len(prediction) == testing_data.shape[0]
    return prediction

def DT_train(train_data, train_labels, **kwargs):
    '''train decision tree classifier

    Args:
    --------
    train_data: train data vector
    train_labels: train data labels

    Returns:
    --------
    clf: DT classifer
    '''
    clf = tree.DecisionTreeClassifier(max_depth=10)
    clf = clf.fit(train_data, train_labels, **kwargs)
    return clf

def DT_pred(clf, testing_data):
    '''prediction by DT classifier

    Args:
    --------
    clf: DT classifier
    testing_data: testing data vector

    Returns:
    --------
    prediction: prediction list
    '''
    prediction = list(clf.predict(testing_data))
    assert len(prediction) == testing_data.shape[0]
    return prediction
