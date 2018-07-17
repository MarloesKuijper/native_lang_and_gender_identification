import os, sys, re, subprocess, shlex, argparse, json
import numpy as np
from sklearn import svm
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict
from sklearn.utils import shuffle
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
    parser.add_argument("-cross","--cross", action="store_true", help="Whether it is cross genre or not")
    parser.add_argument("-g","--gender", action="store_true", help="Whether the type is gender or not")
    parser.add_argument("-m","--medium", required=True, type=str, help="Which medium if within genre")
    parser.add_argument("-name","--name", type=str, help="Model name (filename) to save stacked features to")
    parser.add_argument("-data_file","--data_file", type=str, help="Data file")
    parser.add_argument("-ling_features", "--ling_features", type=str, help="Linguistic feature file")
    parser.add_argument("-second_model", "--second_model", type=str, help="Name second model (pos, posfunc, ling)")
    args = parser.parse_args()
    return args

def preprocess_text(text, affix, country, pos_tagging=False):
    # make sure you avoid counting URL and NUM as capitalized words (for e.g. German)
    # maybe @username?, maybe check again for RTs?
    with open("fileyouwontneed.txt", "a", encoding="utf-8") as outfile:
        tokenizer = TweetTokenizer(reduce_len=True, preserve_case=True)
        cleanr = re.compile('<.*?>')
        remove_markup = re.sub(cleanr, '', text)
        replace_urls = re.sub(r"http\S+", "URL", remove_markup)
        replace_digits = re.sub(r'\d+', "NUM", replace_urls)
        if affix.endswith("@"):
            replace_digits = re.sub(r'\.([a-zA-Z])', r'. \1', replace_digits)
        text = tokenizer.tokenize(replace_digits)
        if pos_tagging:
            original_text = text
            text = nltk.pos_tag(text)
            text = " ".join([nltk.map_tag("en-ptb", "universal", tag) for word, tag in text])
            #text = " ".join([item[1] if item[1].startswith("NN") or item[1].startswith("VB") else item[0] for item in text])

            outfile.write("{0},{1},{2}\n".format(" ".join(original_text)," ".join(text), country))
        else:
            text = " ".join([item for item in text])

        return text
    

def cross_val_cross_genre(dataset1, dataset2, gender=False):
    vec = TfidfVectorizer(lowercase=False, ngram_range=(1,2), analyzer="word") 
    clf = svm.SVC(kernel="linear", C=10)
    clf = Pipeline( [('vec', vec), ('cls', clf)] )
    X_train, X_test, y_train, y_test = split_dataset_cross(dataset1, dataset2, gender, downsample=False)
    X = X_train + X_test
    y = y_train + y_test
    X, y = shuffle(X,y, random_state=42)
    scores = cross_val_score(clf, X, y, cv=10, scoring='f1_macro')
    mean_f1 = np.mean(scores)
    print(mean_f1)

def stacking(train, test, within=False, gender=False):
    ### cross-val for training features > probabilities
    ### predictions based on training data for dev > probabilities
    ### maybe already get test predictions > probabilities too?
    ### do this for both ngrams + linguistic and then add together 9 + 9
    pass


def load_data(data_file, corpus=True):
    datafile = pd.read_csv(data_file, header=0)
    train = datafile.loc[datafile.iloc[:,-1] == "train"]
    test = datafile.loc[datafile.iloc[:,-1] == "test"]
    dev = datafile.loc[datafile.iloc[:,-1] == "dev"]
    if corpus:
        X_train = [item[0] for item in train.iloc[:,:-3].values]
        y_train = train.iloc[:,-3]
        X_dev = [item[0] for item in dev.iloc[:,:-3].values]
        y_dev = dev.iloc[:,-3]
        X_test = [item[0] for item in test.iloc[:,:-3].values]
        y_test = test.iloc[:,-3]
    else:
        X_train = [item for item in train.iloc[:,:-3].values]
        y_train = train.iloc[:,-3]
        X_dev = [item for item in dev.iloc[:,:-3].values]
        y_dev = dev.iloc[:,-3]
        X_test = [item for item in test.iloc[:,:-3].values]
        y_test = test.iloc[:,-3]
    y_extra_train = train.iloc[:,-2]
    y_extra_dev = dev.iloc[:,-2]
    y_extra_test = test.iloc[:,-2]
    ## switch this back on and change numbers to correspondng correct columns (-3 usually), and add the 3 things to the return statement here
    return X_train, y_train, X_dev, y_dev, X_test, y_test, y_extra_train, y_extra_dev, y_extra_test, train, test, dev

def get_ensembled_training_features(model, X_train, y_train, model2, X_train2, y_train2):
    probabilities_m1 = cross_val_predict(model, X_train, y_train, cv=5, method="predict_proba")
    probabilities_m2 = cross_val_predict(model2, X_train2, y_train2, cv=5, method="predict_proba")
    ensembled_mean_feats_train = np.mean(np.array([probabilities_m1, probabilities_m2]), axis=0)
    # OR np.average(np.array([probabilities_m1, probabilities_m2]), axis=0)

    return ensembled_mean_feats_train

def get_ensembled_test_features(model, X_train, y_train, X_test, model2, X_train2, y_train2, X_test2):
    trained_model_m1 = model.fit(X_train, y_train)
    probabilities_test_m1 = trained_model_m1.predict_proba(X_test)

    trained_model_m2 = model2.fit(X_train2, y_train2)
    probabilities_test_m2 = trained_model_2.predict_proba(X_test2)

    ensembled_mean_feats_train = np.mean(np.array([probabilities_test_m1, probabilities_test_m2]), axis=0)
    return ensembled_mean_feats_test


def get_stacked_training_features(model, X_train, y_train, model2, X_train2, y_train2):
    probabilities_m1 = cross_val_predict(model, X_train, y_train, cv=5, method="predict_proba")
    probabilities_m2 = cross_val_predict(model2, X_train2, y_train2, cv=5, method="predict_proba")
    stacked_probs_train = np.hstack((probabilities_m1, probabilities_m2))
    return stacked_probs_train

def get_stacked_test_features(model, X_train, y_train, X_test, model2, X_train2, y_train2, X_test2):
    trained_model_m1 = model.fit(X_train, y_train)
    probabilities_test_m1 = trained_model_m1.predict_proba(X_test)

    trained_model_m2 = model2.fit(X_train2, y_train2)
    probabilities_test_m2 = trained_model_m2.predict_proba(X_test2)

    stacked_probs_test = np.hstack((probabilities_test_m1, probabilities_test_m2))

    return stacked_probs_test

def save_model(probs_train, y_train, y_extra_train, probs_dev, y_dev, y_extra_dev, probs_test, y_test, y_extra_test):
    assert len(probs_train) == len(y_train) and len(probs_dev) == len(y_dev) and len(probs_test) == len(y_test)
    train_dummy = np.array(["train" for item in y_train]).reshape(-1,1)
    dev_dummy = np.array(["dev" for item in y_dev]).reshape(-1,1)
    test_dummy = np.array(["test" for item in y_test]).reshape(-1,1)

    data_train = pd.DataFrame(np.hstack((probs_train, np.array(y_train).reshape(-1,1), np.array(y_extra_train).reshape(-1,1), train_dummy)))
    data_dev = pd.DataFrame(np.hstack((probs_dev, np.array(y_dev).reshape(-1,1), np.array(y_extra_dev).reshape(-1,1), dev_dummy)))
    data_test = pd.DataFrame(np.hstack((probs_test, np.array(y_test).reshape(-1,1), np.array(y_extra_test).reshape(-1,1), test_dummy)))
    frames = [data_train, data_dev, data_test]
    df = pd.concat(frames)
    df.to_csv("stacked_feats_{0}.csv".format(args.name), index=False, header=False)

def get_best_models_for_task(model_task_dict, model="model2"):
    if args.cross:
        if "native_lang" in args.name:
            if args.medium == "twitter":
                model1 = model_task_dict["cross-native-lang-tm"]["model1"]
                model2 = model_task_dict["cross-native-lang-tm"][model]
            else:
                model1 = model_task_dict["cross-native-lang-mt"]["model1"]
                model2 = model_task_dict["cross-native-lang-mt"][model]
        else:
            if args.medium == "twitter":
                model1 = model_task_dict["cross-gender-tm"]["model1"]
                model2 = model_task_dict["cross-gender-tm"][model]
            else:
                model1 = model_task_dict["cross-gender-mt"]["model1"]
                model2 = model_task_dict["cross-gender-mt"][model]
    else:
        if "native_lang" in args.name:
            if args.medium == "twitter":
                model1 = model_task_dict["within-native-lang-twitter"]["model1"]
                model2 = model_task_dict["within-native-lang-twitter"][model]
            else:
                model1 = model_task_dict["within-native-lang-medium"]["model1"]
                model2 = model_task_dict["within-native-lang-medium"][model]
        else:
            if args.medium == "twitter":
                model1 = model_task_dict["within-gender-twitter"]["model1"]
                model2 = model_task_dict["within-gender-twitter"][model]
            else:
                model1 = model_task_dict["within-gender-medium"]["model1"]
                model2 = model_task_dict["within-gender-medium"][model]

    return model1, model2


if __name__ == "__main__":
    args = create_arg_parser()
    #print(args.ff)

    X_train, y_train, X_dev, y_dev, X_test, y_test, y_extra_train, y_extra_dev, y_extra_test, train, test, dev = load_data(args.data_file)

    if args.second_model == "pos":
        X_train_second = [" ".join([item[1] for item in nltk.pos_tag(nltk.word_tokenize(item))]) for item in X_train]
        X_dev_second = [" ".join([item[1] for item in nltk.pos_tag(nltk.word_tokenize(item))]) for item in X_dev]
        X_test_second = [" ".join([item[1] for item in nltk.pos_tag(nltk.word_tokenize(item))]) for item in X_test]
        y_train_second = y_train
    elif args.second_model == "posfunc":
        X_train_second = [" ".join([item[1] if item[1].startswith("NN") or item[1].startswith("VB") else item[0] for item in nltk.pos_tag(nltk.word_tokenize(item))]) for item in X_train]
        X_dev_second = [" ".join([item[1] if item[1].startswith("NN") or item[1].startswith("VB") else item[0] for item in nltk.pos_tag(nltk.word_tokenize(item))]) for item in X_dev]
        X_test_second = [" ".join([item[1] if item[1].startswith("NN") or item[1].startswith("VB") else item[0] for item in nltk.pos_tag(nltk.word_tokenize(item))]) for item in X_test]
        y_train_second = y_train
    elif args.second_model == "ling":
        ## hier de drie extra dingen toevoegen
        X_train_second, y_train_second, X_dev_second, y_dev_second, X_test_second, y_test_second, y_extra_train_second, y_extra_dev_second, y_extra_test_second, train, test, dev = load_data(args.ling_features, corpus=False)

        ## make sure this is identical!!
        # print(y_train[:5])
        # print(ling_y_train[:5])
        # print(y_test[:5])
        # print(ling_y_test[:5])
        if args.cross:
            if args.gender:
                ## use color
                X_train_second = train[["color_regular", "color_special"]]
                X_dev_second = dev[["color_regular", "color_special"]]
                X_test_second = test[["color_regular", "color_special"]]
            else:
                if args.medium == "twitter":
                    ## use swearwords
                    X_train_second = train[["swearwords"]]
                    X_dev_second = dev[["swearwords"]]
                    X_test_second = test[["swearwords"]]
                else:
                    ## use all gender
                    X_train_second = train[["filler", "hedges", "swearwords", "sentiment", "min_response", "adjectives", "color_special", "color_regular","punct_?", "punct_!"]]
                    X_dev_second = dev[["filler", "hedges", "swearwords", "sentiment", "min_response", "adjectives", "color_special", "color_regular","punct_?", "punct_!"]]
                    X_test_second = test[["filler", "hedges", "swearwords", "sentiment", "min_response", "adjectives", "color_special", "color_regular","punct_?", "punct_!"]]

        else:
            if args.gender:
                ## use all 13
                X_train_second = train[["sent_length", "capitalization", "articles", "mult_negs", "filler", "hedges", "swearwords", "sentiment", "min_response", "adjectives","has_svo", "has_sov", "has_vos", "has_vso", "has_ovs", "has_osv", "color_special", "color_regular","punct_?", "punct_!"]]
                X_dev_second = dev[["sent_length", "capitalization", "articles", "mult_negs", "filler", "hedges", "swearwords", "sentiment", "min_response", "adjectives","has_svo", "has_sov", "has_vos", "has_vso", "has_ovs", "has_osv", "color_special", "color_regular","punct_?", "punct_!"]]
                X_test_second = test[["sent_length", "capitalization", "articles", "mult_negs", "filler", "hedges", "swearwords", "sentiment", "min_response", "adjectives","has_svo", "has_sov", "has_vos", "has_vso", "has_ovs", "has_osv", "color_special", "color_regular","punct_?", "punct_!"]]

            else:
                if args.medium == "twitter":
                    # use all 13
                    X_train_second = train[["sent_length", "capitalization", "articles", "mult_negs", "filler", "hedges", "swearwords", "sentiment", "min_response", "adjectives","has_svo", "has_sov", "has_vos", "has_vso", "has_ovs", "has_osv", "color_special", "color_regular","punct_?", "punct_!"]]
                    X_dev_second = dev[["sent_length", "capitalization", "articles", "mult_negs", "filler", "hedges", "swearwords", "sentiment", "min_response", "adjectives","has_svo", "has_sov", "has_vos", "has_vso", "has_ovs", "has_osv", "color_special", "color_regular","punct_?", "punct_!"]]
                    X_test_second = test[["sent_length", "capitalization", "articles", "mult_negs", "filler", "hedges", "swearwords", "sentiment", "min_response", "adjectives","has_svo", "has_sov", "has_vos", "has_vso", "has_ovs", "has_osv", "color_special", "color_regular","punct_?", "punct_!"]]
                    
                else:
                    #all native lang
                    X_train_second = train[["sent_length", "capitalization", "articles", "mult_negs", "sentiment","has_svo", "has_sov"]]
                    X_dev_second = dev[["sent_length", "capitalization", "articles", "mult_negs", "sentiment","has_svo", "has_sov"]]
                    X_test_second = test[["sent_length", "capitalization", "articles", "mult_negs", "sentiment","has_svo", "has_sov"]]

    # with open("best_models_per_task_ngram_posfunc.json", "w") as fp:
    #     json.dump(best_models_per_task_ngram_posfunc, fp)
    with open("best_models_per_task_{0}.json".format(args.second_model), "r") as fp:
        best_models_per_task = json.load(fp)
    
    print(best_models_per_task)


    clf_ngrams_params, clf_second_model_params = get_best_models_for_task(best_models_per_task)


    # ## models:
    clf_ngrams = Pipeline([("vect", TfidfVectorizer(ngram_range=clf_ngrams_params["ngram_range"], analyzer=clf_ngrams_params["analyzer"], lowercase=False)), ("clf", svm.SVC(C=clf_ngrams_params["C"], kernel=clf_ngrams_params["kernel"], probability=True))])
    print(clf_ngrams.get_params())
    if args.second_model == "ling":
        second_clf = svm.SVC(C=clf_second_model_params["C"], kernel=clf_second_model_params["kernel"], probability=True)
    else:
        second_clf = Pipeline([("vect", TfidfVectorizer(ngram_range=clf_second_model_params["ngram_range"], analyzer=clf_second_model_params["analyzer"], lowercase=False)), ("clf", svm.SVC(C=clf_second_model_params["C"], kernel=clf_second_model_params["kernel"],probability=True))])


    if args.second_model == "ling" and args.cross and args.gender and args.medium == "twitter":
        # best features are POS/FUNCTION features
        X_train  = [nltk.word_tokenize(item) for item in X_train]
        X_dev = [nltk.word_tokenize(item) for item in X_dev]
        X_test = [nltk.word_tokenize(item) for item in X_test]
        X_train = [" ".join([item[1] if item[1].startswith("NN") or item[1].startswith("VB") else item[0] for item in nltk.pos_tag(text)]) for text in X_train]
        X_dev = [" ".join([item[1]  if item[1].startswith("NN") or item[1].startswith("VB") else item[0] for item in nltk.pos_tag(text) ]) for text in X_dev]
        X_test = [" ".join([item[1] if item[1].startswith("NN") or item[1].startswith("VB") else item[0] for item in nltk.pos_tag(text) ]) for text in X_test]



    stacked_training_feats = get_stacked_training_features(clf_ngrams, X_train, y_train, second_clf, X_train_second, y_train_second)
    stacked_dev_feats = get_stacked_test_features(clf_ngrams, X_train, y_train, X_dev, second_clf, X_train_second, y_train_second, X_dev_second)
    stacked_test_feats = get_stacked_test_features(clf_ngrams, X_train, y_train, X_test, second_clf, X_train_second, y_train_second, X_test_second)
    print(len(stacked_training_feats[0]))


    clf = svm.SVC(C=1, kernel="linear")
    clf.fit(stacked_training_feats, y_train)
    preds = clf.predict(stacked_dev_feats)
    print("Stacking Ngrams + {0}...".format(args.second_model))
    print("Predictions stacked for file {0}".format(args.data_file))
    print(accuracy_score(y_dev, preds))

    save_model(stacked_training_feats, y_train, y_extra_train, stacked_dev_feats, y_dev, y_extra_dev, stacked_test_feats, y_test, y_extra_test)

    ### get ALL Data not just :400!!!!!