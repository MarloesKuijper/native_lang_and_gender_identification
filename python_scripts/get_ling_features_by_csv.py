import os, sys, re, subprocess, shlex, argparse
import numpy as np
from sklearn import svm
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.utils import shuffle, validation
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, classification_report, accuracy_score, f1_score
from collections import defaultdict,Counter
import math
import pandas as pd 
from nltk.tokenize import TweetTokenizer, word_tokenize, sent_tokenize
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB
#import matplotlib.pyplot as plt
import nltk
import spacy
from textblob import TextBlob

def create_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-l","--length", required=True, type=str, help="Length of each document")
    parser.add_argument("-c","--countries", type=str, help="Which countries (languages) to use")
    parser.add_argument("-p","--path", type=str, help="Where the twitter and medium folders are located")
    parser.add_argument("-cross","--cross", action="store_true", help="Whether it is cross genre or not")
    parser.add_argument("-g","--gender", action="store_true", help="Whether the type is gender or not")
    parser.add_argument("-m","--medium", type=str, help="Which medium if within genre")
    parser.add_argument("-name","--name", type=str, help="Model name (filename)")
    parser.add_argument("-output_type","--output_type", type=str, help="Output type")
    parser.add_argument("-data_file","--data_file", type=str, help="Data file")
    parser.add_argument("-ff", "--ff", type=str, help="Feature file")
    args = parser.parse_args()
    return args

def load_data(data_file, load_data=True, header_present=[0]):
    datafile = pd.read_csv(data_file, header=header_present)
    train = datafile.loc[datafile.iloc[:,-1] == "train"]
    test = datafile.loc[datafile.iloc[:,-1] == "test"]
    dev = datafile.loc[datafile.iloc[:,-1] == "dev"]
    
    return train, test, dev

def test_svm_params(model_name=""):
    # load linguistic features file (per task) as dataframe
    # then separate dataframe into the specific features that you want to test (13 + all)
    # run the bottom part

    train, test, dev = load_data(args.data_file, header_present=[0])



    column_values = ["sent_length", "capitalization", "articles", "mult_negs", "filler", "hedges", "swearwords", "sentiment", "min_response", "adjectives"]
    columns_clustered = [["has_svo", "has_sov", "has_vos", "has_vso", "has_ovs", "has_osv"], ["color_special", "color_regular"],["punct_?", "punct_!"]]
    columns_all = [["sent_length", "capitalization", "articles", "mult_negs", "filler", "hedges", "swearwords", "sentiment", "min_response", "adjectives","has_svo", "has_sov", "has_vos", "has_vso", "has_ovs", "has_osv", "color_special", "color_regular","punct_?", "punct_!"]]


    ## when doing columns clustered: maybe change X_train, X_dev etc., change model_name to incorporate the correct feature name (feature is in this case a list!)
    ## column_clusters > enumerate > index = 0 = word_order, index = 1 = color, index = 2 = punct
    ## when doing combined features > just run columns all and change the same stuff as with columns_clustered


    column_values = [["sent_length", "capitalization", "articles", "mult_negs", "sentiment","has_svo", "has_sov"], ["filler", "hedges", "swearwords", "sentiment", "min_response", "adjectives", "color_special", "color_regular","punct_?", "punct_!"]]

    feature_names = [ "all_native_lang_feats_together","all_gender_feats_together"]

    for ix, feature in enumerate(column_values):
            X_train = train[feature]
            X_dev = dev[feature]
            X_test = test[feature]
            y_train = np.array(train[["y_label"]]).ravel()
            y_dev = np.array(dev[["y_label"]]).ravel()
            y_test = np.array(test[["y_label"]]).ravel()
            kernels = ["linear", "rbf"]
            Cs = [1,10,20]
            results = []
            model_file_name = model_name[:-4] + "_" + feature_names[ix] + ".txt" 
            with open(model_file_name, "w", encoding="utf-8") as outfile:
                # for ngram_range in ngram_ranges:
                #     for analyzer in analyzers:
                for kernel in kernels:
                    for C in Cs:
                        #vec = TfidfVectorizer(lowercase=False, ngram_range=ngram_range, analyzer=analyzer) 
                        clf = svm.SVC(kernel=kernel, C=C)
                        #clf = Pipeline( [('vec', vec), ('cls', clf)] )
                        # if feature != "combined_feats_all13":
                        #     X_train = np.array(X_train).reshape(-1, 1)
                        #     X_dev = np.array(X_dev).reshape(-1,1)
                        #     X_test = np.array(X_test).reshape(-1,1)
                        clf.fit(X_train, y_train)
                        y_guess = clf.predict(X_dev)
                        accuracy = accuracy_score(y_dev, y_guess)
                        f1score = f1_score(y_dev, y_guess, average="macro")
                        outfile.write("SVM with kernel {0}, C {1}\n".format(kernel, C))
                        outfile.write("Accuracy: {0}\n".format(accuracy))
                        outfile.write("(Macro) F1-score: {0}\n".format(f1score))
                        outfile.write(classification_report(y_dev, y_guess))
                        outfile.write("\n\n")
                        results.append(("SVM with kernel {0}, C {1}\n".format(kernel, C), f1score, accuracy))

            if args.cross:
                ordered_name = "./ORDERED_{0}_{1}.txt".format(args.name[:-4], feature_names[ix])
            else:
                if args.medium == "twitter":
                    ordered_name = "./ORDERED_{0}_{1}.txt".format(args.name[:-4], feature_names[ix])
                else:
                    ordered_name = "./ORDERED_{0}_{1}.txt".format(args.name[:-4], feature_names[ix])
            print(ordered_name)
            with open(ordered_name, "w", encoding="utf-8") as out:
                ordered_f1 = sorted(results, key=lambda x: x[1], reverse=True)
                ordered_acc = sorted(results, key=lambda x: x[2], reverse=True)
                out.write("F1-scores descending order\n")
                for item in ordered_f1:
                    out.write(item[0])
                    out.write("F1-score: " + str(item[1]) + "\n")
                    out.write("Accuracy: " + str(item[2]) + "\n")
                    out.write("\n")
                out.write("\n")
                out.write("Accuracy descending order\n")
                for item in ordered_acc:
                    out.write(item[0])
                    out.write("Accuracy: " + str(item[2]) + "\n")
                    out.write("F1-score: " + str(item[1]) + "\n")
                    out.write("\n")


if __name__ == "__main__":
    args = create_arg_parser()

    if args.cross:   
        test_svm_params(model_name=args.name)
    else:

        test_svm_params(model_name=args.name)
        