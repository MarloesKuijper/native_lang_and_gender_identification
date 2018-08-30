from nltk.tokenize import TweetTokenizer, word_tokenize, sent_tokenize
import argparse, os
from collections import defaultdict
import pandas as pd
import numpy as np
import json, re
from numpy import array
from numpy import asarray
from numpy import zeros
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
np.random.seed(42)

def create_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-name","--name", type=str, help="Model name (filename)")
    parser.add_argument("-data_file","--data_file", type=str, help="Data file")
    parser.add_argument("-preds_test_single", "--preds_test_single", type=str, help="Predictions test single run")
    parser.add_argument("-preds_dev_ten", "--preds_dev_ten", type=str, help="Predictions dev 10 runs")
    parser.add_argument("-preds_svm_d", "--preds_svm_d", type=str, help="Predictions SVM 10 runs dev")
    parser.add_argument("-preds_svm_t", "--preds_svm_t", type=str, help="Predictions SVM 10 runs test")
    parser.add_argument("-preds_neural_d", "--preds_neural_d", type=str, help="Predictions Neural 10 runs dev")
    parser.add_argument("-preds_neural_t", "--preds_neural_t", type=str, help="Predictions Neural 10 runs test")
    args = parser.parse_args()
    return args

def load_data(df, load_data=True, header_present=[0]):
    datafile = pd.read_csv(df, header=header_present)
    train = datafile.loc[datafile.iloc[:,-1] == "train"]
    test = datafile.loc[datafile.iloc[:,-1] == "test"]
    dev = datafile.loc[datafile.iloc[:,-1] == "dev"]
    X_train = [item[0] for item in train.iloc[:,:-3].values]
    y_train = train.iloc[:,-3]
    X_dev = [item[0] for item in dev.iloc[:,:-3].values]
    y_dev = dev.iloc[:,-3]
    X_test = [item[0] for item in test.iloc[:,:-3].values]
    y_test = test.iloc[:,-3]
    
    return X_train, y_train, X_dev, y_dev, X_test, y_test

def get_accuracy(preds, y_test):
    if "native_lang" in args.name:
        classes = ["germany", "iran", "italy", "new-delhi", "poland", "portugal", "russia","spain", "the-netherlands"]
    else:
        classes = ["female", "male"]
    preds = np.load(preds)
    argmax_preds = np.argmax(preds, axis=1)
    preds_to_cat = [classes[item] for item in argmax_preds]
    #print(np.argmax(preds, axis=1))
    print(accuracy_score(y_test, preds_to_cat))


def create_ensemble_neural_svm(preds_neural, preds_svm, y_test):
    if "native_lang" in args.name:
        classes = ["germany", "iran", "italy", "new-delhi", "poland", "portugal", "russia","spain", "the-netherlands"]
    else:
        classes = ["female", "male"]
    preds_neural = np.load(preds_neural)
    preds_svm = np.load(preds_svm)
    predictions = [preds_neural, preds_svm]
    ensemble_preds = [list(np.mean(item, axis=0)) for item in list(zip(*predictions))]
    argmax_preds = np.argmax(ensemble_preds, axis=1)
    preds_to_cat = [classes[item] for item in argmax_preds]
    print(accuracy_score(y_test, preds_to_cat))

if __name__ == "__main__":
    args = create_arg_parser()

    X_train, y_train, X_dev, y_dev, X_test, y_test = load_data(args.data_file, header_present=[0])
    print(args.name)
    print("Accuracy test set, single run:")
    get_accuracy(args.preds_test_single, y_test)
    print("Accuracy dev set, averaged over 10 runs:")
    get_accuracy(args.preds_dev_ten, y_dev)

    print("Accuracy dev set ensemble neural + SVM (10 runs): ")
    create_ensemble_neural_svm(args.preds_dev_ten, args.preds_svm_d, y_dev)

    print("Accuracy test set ensemble neural + SVM (10 runs): ")
    create_ensemble_neural_svm(args.preds_neural_t, args.preds_svm_t, y_test)







