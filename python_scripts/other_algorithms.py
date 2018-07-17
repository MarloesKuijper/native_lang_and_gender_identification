import os, sys, re, subprocess, shlex, argparse
import numpy as np
from sklearn import svm
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.utils import shuffle
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, classification_report, accuracy_score, f1_score
from collections import defaultdict,Counter
import math
import pandas as pd 
from nltk.tokenize import TweetTokenizer
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
#import seaborn as sns
#import matplotlib.pyplot as plt
import nltk

def create_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-cross","--cross", action="store_true", help="Whether it is cross genre or not")
    parser.add_argument("-g","--gender", action="store_true", help="Whether the type is gender or not")
    parser.add_argument("-name","--name", type=str, help="Model name (filename)")
    parser.add_argument("-ff","--ff", type=str, help="Feature file (pos distro features")
    args = parser.parse_args()
    return args

def load_data(data_file, load_data=True, header_present=[0]):
    datafile = pd.read_csv(data_file, header=header_present)
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
    


def test_params(gender=False, within=False,model_name=""):
    X_train, y_train, X_dev, y_dev, X_test, y_test = load_data(args.ff, header_present=[0])

    random_state = 2


    classifiers = []
    classifiers.append(svm.SVC(random_state=random_state))
    classifiers.append(DecisionTreeClassifier(random_state=random_state))
    classifiers.append(AdaBoostClassifier(DecisionTreeClassifier(random_state=random_state),random_state=random_state,learning_rate=0.1))
    classifiers.append(RandomForestClassifier(random_state=random_state))
    classifiers.append(ExtraTreesClassifier(random_state=random_state))
    classifiers.append(GradientBoostingClassifier(random_state=random_state))
    classifiers.append(MLPClassifier(random_state=random_state))
    classifiers.append(KNeighborsClassifier())
    classifiers.append(LogisticRegression(random_state = random_state))
    #classifiers.append(LinearDiscriminantAnalysis())

    cv_results = []
    for classifier in classifiers :
        pipeline = Pipeline([("vect", TfidfVectorizer(ngram_range=(1,2), lowercase=False)), ("clf", classifier)])
        pipeline.fit(X_train,y_train)
        preds = pipeline.predict(X_dev)
        accuracy = accuracy_score(y_dev, preds)
        cv_results.append(accuracy)

    
    cv_res = pd.DataFrame({"CrossValMeans":cv_results,"Algorithm":["SVC","DecisionTree","AdaBoost",
    "RandomForest","ExtraTrees","GradientBoosting","MultipleLayerPerceptron","KNeighbours","LogisticRegression"]})

    cv_res.to_csv("results_devset_{0}".format(args.name))
    print(cv_res)

def test_specific_params(gender=False, within=False,model_name=""):
    X_train, y_train, X_dev, y_dev, X_test, y_test = load_data(args.ff, header_present=[0])

    random_state = 2
    ngram_ranges = [(1,1),(1,2), (1,3),(1,4),(1,5), (1,6),(1,7),(1,8),(2,3)]
    analyzers = ["word", "char"]
    max_depths = [5, 10]
    min_samples_leafs = [5, 10, 20]
    no_estimators = [100, 300, 500]
    n_neighbors = [20]
    weights = ["uniform", "distance"]

    dt_ngram_col = []
    dt_analyzer_col = []
    dt_depth_col = []
    dt_min_samples_col = []
    dt_acc_col = []

    gb_ngram_col = []
    gb_analyzer_col = []
    gb_est_col = []
    gb_depth_col = []
    gb_min_samples_col = []
    gb_acc_col = []

    knn_ngram_col = []
    knn_analyzer_col = []
    knn_neighbors_col = []
    knn_weights_col = []
    knn_acc_col = []


    cv_results = []
    with open("other_algorithms_results_{0}_decision_tree.txt".format(args.name), "w", encoding="utf-8") as outfile:
        ngram_ranges_dt = [(1,1),(1,2), (1,3), (1,6),(1,7),(1,8)]
        for ngram_range in ngram_ranges_dt:
            for analyzer in analyzers:
                    for depth in max_depths:
                        for min_samples in min_samples_leafs:
                            clf = DecisionTreeClassifier(random_state=2, max_depth=depth, min_samples_leaf=min_samples)
                            pipeline = Pipeline([("vect", TfidfVectorizer(lowercase=False, ngram_range=ngram_range, analyzer=analyzer)), ("clf", clf)])
                            print("fitting data dt")
                            pipeline.fit(X_train,y_train)
                            preds = pipeline.predict(X_dev)
                            accuracy = accuracy_score(y_dev, preds)
                            print("Decision Tree Accuracy {0}, with ngrams {1}, analyzer {2}, max_depth {3}, min samples leaf {4}\n".format(accuracy, ngram_range, analyzer, depth, min_samples))
                            dt_ngram_col.append(ngram_range)
                            dt_analyzer_col.append(analyzer)
                            dt_depth_col.append(depth)
                            dt_min_samples_col.append(min_samples)
                            dt_acc_col.append(accuracy)
                            outfile.write("Decision Tree Accuracy {0}, with ngrams {1}, analyzer {2}, max_depth {3}, min samples leaf {4}\n".format(accuracy, ngram_range, analyzer, depth, min_samples))
    
    df_decision_tree = pd.DataFrame({"ngram_range": dt_ngram_col, "analyzer": dt_analyzer_col, "max_depth": dt_depth_col, "min_samples_leaf": dt_min_samples_col, "accuracy": dt_acc_col})
    df_decision_tree.to_csv("results_devset_decision_tree_{0}.csv".format(args.name))

    with open("other_algorithms_results_{0}_knn.txt".format(args.name), "w", encoding="utf-8") as outfile:
        analyzers = ["word"]
        for ngram_range in ngram_ranges:
            for analyzer in analyzers:
                for neighbor in n_neighbors:
                    for weight in weights:
                        clf = KNeighborsClassifier(n_neighbors=neighbor, weights=weight)
                        pipeline = Pipeline([("vect", TfidfVectorizer(lowercase=False, ngram_range=ngram_range, analyzer=analyzer)), ("clf", clf)])
                        print("fitting data knn")
                        pipeline.fit(X_train,y_train)
                        preds = pipeline.predict(X_dev)
                        accuracy = accuracy_score(y_dev, preds)
                        print("KNN Accuracy {0}, with ngrams {1}, analyzer {2}, neighbors {3}, weight {4}\n".format(accuracy, ngram_range, analyzer, neighbor, weight))
                        knn_ngram_col.append(ngram_range)
                        knn_analyzer_col.append(analyzer)
                        knn_neighbors_col.append(neighbor)
                        knn_weights_col.append(weight)
                        knn_acc_col.append(accuracy)
                        outfile.write("KNN Accuracy {0}, with ngrams {1}, analyzer {2}, neighbors {3}, weight {4}\n".format(accuracy, ngram_range, analyzer, neighbor, weight))

    df_knn = pd.DataFrame({"ngram_range": knn_ngram_col, "analyzer": knn_analyzer_col, "neighbors": knn_neighbors_col, "weights": knn_weights_col, "accuracy": knn_acc_col})
    df_knn.to_csv("results_devset_knn_{0}.csv".format(args.name))
                    # elif i == 1:
                    #     pass
                        # for est in no_estimators:
                        #     for depth in max_depths:
                        #         for min_samples in min_samples_leafs:
                        #             clf = GradientBoostingClassifier(max_depth=depth, min_samples_leaf=min_samples, n_estimators=est, random_state=2)
                        #             pipeline = Pipeline([("vect", TfidfVectorizer(lowercase=False, ngram_range=ngram_range, analyzer=analyzer)), ("clf", clf)])
                        #             print("fitting data gb")
                        #             pipeline.fit(X_train,y_train)
                        #             preds = pipeline.predict(X_dev)
                        #             accuracy = accuracy_score(y_dev, preds)
                        #             print("accuracy: {0}".format(accuracy))
                        #             gb_ngram_col.append(ngram_range)
                        #             gb_analyzer_col.append(analyzer)
                        #             gb_est_col.append(est)
                        #             gb_depth_col.append(depth)
                        #             gb_min_samples_col.append(min_samples)
                        #             gb_acc_col.append(accuracy)
                        #             outfile.write("Gradient Boosting Accuracy {0}, with ngrams {1}, analyzer {2}, max_depth {3}, min samples leaf {4}, number estimators {5}\n".format(accuracy, ngram_range, analyzer, depth, min_samples, est))
                        

    
    # df_decision_tree = pd.DataFrame({"ngram_range": dt_ngram_col, "analyzer": dt_analyzer_col, "max_depth": dt_depth_col, "min_samples_leaf": dt_min_samples_col, "accuracy": dt_acc_col})
    # df_gradient_boost = pd.DataFrame({"ngram_range": gb_ngram_col, "analyzer": gb_analyzer_col, "max_depth": gb_depth_col, "min_samples_leaf": gb_min_samples_col, "estimators": gb_est_col, "accuracy": gb_acc_col})
    # df_knn = pd.DataFrame({"ngram_range": knn_ngram_col, "analyzer": knn_analyzer_col, "neighbors": knn_neighbors_col, "weights": knn_weights_col, "accuracy": knn_acc_col})

    
    #df_gradient_boost.to_csv("results_devset_gradient_boost_{0}.csv".format(args.name))
    
    

if __name__ == "__main__":
    args = create_arg_parser()
    #print(args.ff)

    ## GRID SEARCH
    if args.cross:
        test_specific_params(gender=args.gender, within=False, model_name=args.name)
    else:
        test_specific_params(gender=args.gender, within=True, model_name=args.name)
     



