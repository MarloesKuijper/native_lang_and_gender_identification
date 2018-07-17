import os, sys, re, subprocess, shlex, argparse
import numpy as np
from sklearn import svm
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict
from sklearn.utils import shuffle
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, classification_report, accuracy_score, f1_score
from collections import defaultdict,Counter
from sklearn.preprocessing import LabelEncoder
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
from sklearn.base import BaseEstimator, TransformerMixin
#import matplotlib.pyplot as plt
import nltk
import spacy
from textblob import TextBlob
from scipy.sparse import hstack as sphstack

def create_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-cross","--cross", action="store_true", help="Whether it is cross genre or not")
    parser.add_argument("-g","--gender", action="store_true", help="Whether the type is gender or not")
    parser.add_argument("-m","--medium", required=True, type=str, help="Which medium if within genre")
    parser.add_argument("-name","--name", type=str, help="Model name (filename)")
    parser.add_argument("-data_file","--data_file", type=str, help="Data file")
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
    

def get_best_model_for_task(model_task_dict):
    if args.cross:
        if "native_lang" in args.name:
            if args.medium == "twitter":
                model1 = model_task_dict["cross-native-lang-tm"]["model1"]
            else:
                model1 = model_task_dict["cross-native-lang-mt"]["model1"]
        else:
            if args.medium == "twitter":
                model1 = model_task_dict["cross-gender-tm"]["model1"]
            else:
                model1 = model_task_dict["cross-gender-mt"]["model1"]
    else:
        if "native_lang" in args.name:
            if args.medium == "twitter":
                model1 = model_task_dict["within-native-lang-twitter"]["model1"]
            else:
                model1 = model_task_dict["within-native-lang-medium"]["model1"]
        else:
            if args.medium == "twitter":
                model1 = model_task_dict["within-gender-twitter"]["model1"]
            else:
                model1 = model_task_dict["within-gender-medium"]["model1"]

    return model1

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


def load_data(data_file, stacked=False):
    if stacked:
        datafile = pd.read_csv(data_file, header=None)
    else:
        datafile = pd.read_csv(data_file, header=0)
    train = datafile.loc[datafile.iloc[:,-1] == "train"]
    test = datafile.loc[datafile.iloc[:,-1] == "test"]
    dev = datafile.loc[datafile.iloc[:,-1] == "dev"]
    if stacked:
        X_train = [item for item in train.iloc[:,:-3].values]
        X_dev = [item for item in dev.iloc[:,:-3].values]
        X_test = [item for item in test.iloc[:,:-3].values]
    else:
        X_train = [item[0] for item in train.iloc[:,:-3].values]
        X_dev = [item[0] for item in dev.iloc[:,:-3].values]
        X_test = [item[0] for item in test.iloc[:,:-3].values]
    
    y_train = train.iloc[:,-3]
    y_dev = dev.iloc[:,-3]
    y_test = test.iloc[:,-3]
    y_extra_train = train.iloc[:,-2]
    y_extra_dev = dev.iloc[:,-2]
    y_extra_test = test.iloc[:,-2]
    return X_train, y_train, X_dev, y_dev, X_test, y_test, y_extra_train, y_extra_dev, y_extra_test
    

class GenderTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, keys):
        self.keys = keys

    def fit(self, x, y=None):
        return self

    def transform(self, df):
        return df[self.keys].values

class VarSelect(BaseEstimator, TransformerMixin):
    def __init__(self, keys):
        self.keys = keys
    def fit(self, x, y=None):
        return self
    def transform(self, df):
        return df[self.keys].values

def make_predictions_nl(clf_params, X_train, y_train, X_dev, y_dev, y_extra_train_feat=False, y_extra_dev_feat=False, stacked=True):
    if stacked:
        clf = svm.SVC(C=1, kernel="linear")
        #vect = TfidfVectorizer(ngram_range=(1,2), analyzer="word", lowercase=False)
    else:
        clf = svm.SVC(C=clf_params["C"], kernel=clf_params["kernel"])
        vect = TfidfVectorizer(ngram_range=clf_params["ngram_range"], analyzer=clf_params["analyzer"], lowercase=False)
    if y_extra_train_feat:
        labelencoder = LabelEncoder()
        gender_train = labelencoder.fit_transform(y_extra_train_feat)
        gender_dev = labelencoder.transform(y_extra_dev_feat)
        #print(gender_dev)

        df = pd.DataFrame({"text": X_train, "gender_train": gender_train})
        #print(np.array(X_train).shape, np.array(gender_train).reshape(-1,1).shape)
        if not stacked:
            X_train = vect.fit_transform(X_train)
            X_dev = vect.transform(X_dev)

            X_train_more = sphstack((X_train, df["gender_train"][:, None]))
            
            X_dev_more = sphstack((X_dev, gender_dev[:, None]))
        else:
            # train_labels = np.array(gender_train).reshape(-1,1)
            # dev_labels = np.array(gender_dev).reshape(-1,1)
            X_train_more_new = []
            X_dev_more_new = []
            for ix, item in enumerate(X_train):
                item = list(item)
                item.append(gender_train[ix])
                X_train_more_new.append(item)
            for ix, item in enumerate(X_dev):
                item = list(item)
                item.append(gender_dev[ix])
                #print(gender_dev[ix])
                X_dev_more_new.append(item)
            #print(X_dev_more_new)
            X_train_more = X_train_more_new
            X_dev_more = X_dev_more_new

        clf.fit(X_train_more, y_train)
        predictions = clf.predict(X_dev_more)
        acc_score = accuracy_score(y_dev, predictions)
        f_score = f1_score(y_dev, predictions, average="macro")
        print("Accuracy: {0}".format(acc_score))
        print("F1-score: {0}".format(f_score))
    else:
        # not supplying gender as extra feature here
        if stacked:
            pipeline = Pipeline([("clf", clf)])
        else:
            pipeline = Pipeline([("vect", vect), ("clf", clf)])
        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_dev)
        acc_score = accuracy_score(y_dev, predictions)
        f_score = f1_score(y_dev, predictions, average="macro")
        print("Accuracy: {0}".format(acc_score))
        print("F1-score: {0}".format(f_score))

def get_data_list(X_train, y_train, y_extra_train, country):
    return [X_train[ix] for ix, item in enumerate(y_extra_train) if item == country], [y_train[ix] for ix, item in enumerate(y_extra_train) if item == country]

def get_equal_lists_nl(*nargs, min_value):
    new_datasets = []
    for item in nargs:
        new_datasets.append(item[:min_value])
    
    return new_datasets

def predict_gender(clf_params, X_train, y_train, X_dev, y_dev, y_extra_train, y_extra_dev, downsized=False):
    ## in this case NL is first followed by GENDER in the csv
    ## questions: crossval or dev set?, filter X_dev and y_dev by gender as well or leave as is?

    X_train_nl, y_train_nl = get_data_list(X_train, y_train, y_extra_train, "the-netherlands")
    X_train_de, y_train_de = get_data_list(X_train, y_train, y_extra_train, "germany")
    X_train_es, y_train_es = get_data_list(X_train, y_train, y_extra_train, "spain")
    X_train_it, y_train_it = get_data_list(X_train, y_train, y_extra_train, "italy")
    X_train_pt, y_train_pt = get_data_list(X_train, y_train, y_extra_train, "portugal")
    X_train_ru, y_train_ru = get_data_list(X_train, y_train, y_extra_train, "russia")
    X_train_po, y_train_po = get_data_list(X_train, y_train, y_extra_train, "poland") 
    X_train_pr, y_train_pr = get_data_list(X_train, y_train, y_extra_train, "iran")
    X_train_hi, y_train_hi = get_data_list(X_train, y_train, y_extra_train, "new-delhi")

    X_dev_nl, y_dev_nl = get_data_list(X_dev, y_dev, y_extra_dev, "the-netherlands")
    X_dev_de, y_dev_de = get_data_list(X_dev, y_dev, y_extra_dev, "germany")
    X_dev_es, y_dev_es = get_data_list(X_dev, y_dev, y_extra_dev, "spain")
    X_dev_it, y_dev_it = get_data_list(X_dev, y_dev, y_extra_dev, "italy")
    X_dev_pt, y_dev_pt = get_data_list(X_dev, y_dev, y_extra_dev, "portugal")
    X_dev_ru, y_dev_ru = get_data_list(X_dev, y_dev, y_extra_dev, "russia")
    X_dev_po, y_dev_po = get_data_list(X_dev, y_dev, y_extra_dev, "poland") 
    X_dev_pr, y_dev_pr = get_data_list(X_dev, y_dev, y_extra_dev, "iran")
    X_dev_hi, y_dev_hi = get_data_list(X_dev, y_dev, y_extra_dev, "new-delhi")

    if downsized:
        training_sets = [len(X_train_nl), len(X_train_de), len(X_train_es), len(X_train_it), len(X_train_pt), len(X_train_ru), len(X_train_po), len(X_train_pr), len(X_train_hi)]
        dev_sets = [len(X_dev_nl), len(X_dev_de), len(X_dev_es), len(X_dev_it), len(X_dev_pt), len(X_dev_ru), len(X_dev_po), len(X_dev_pr), len(X_dev_hi)]
        min_val_train = min(training_sets)
        print(training_sets)
        X_train_nl, y_train_nl, X_train_de, y_train_de, X_train_es, y_train_es, X_train_it, y_train_it, X_train_pt, y_train_pt, X_train_ru, y_train_ru, X_train_po, y_train_po, X_train_pr, y_train_pr, X_train_hi, y_train_hi = get_equal_lists_nl(X_train_nl, y_train_nl, X_train_de, y_train_de, X_train_es, y_train_es, X_train_it, y_train_it, X_train_pt, y_train_pt, X_train_ru, y_train_ru, X_train_po, y_train_po, X_train_pr, y_train_pr, X_train_hi, y_train_hi, min_value=min_val_train)
        print(len(X_train_nl), len(X_train_de), len(X_train_es), len(X_train_it), len(X_train_pt), len(X_train_hi))
        min_val_dev = min(dev_sets)
        X_dev_nl, y_dev_nl, X_dev_de, y_dev_de, X_dev_es, y_dev_es, X_dev_it, y_dev_it, X_dev_pt, y_dev_pt, X_dev_ru, y_dev_ru, X_dev_po, y_dev_po, X_dev_pr, y_dev_pr, X_dev_hi, y_dev_hi = get_equal_lists_nl(X_dev_nl, y_dev_nl, X_dev_de, y_dev_de, X_dev_es, y_dev_es, X_dev_it, y_dev_it, X_dev_pt, y_dev_pt, X_dev_ru, y_dev_ru, X_dev_po, y_dev_po, X_dev_pr, y_dev_pr, X_dev_hi, y_dev_hi, min_value=min_val_dev)
        print(dev_sets)
        print(len(X_dev_nl), len(X_dev_de), len(X_dev_es), len(X_dev_it), len(X_dev_pt), len(X_dev_hi))

    
    assert len(X_train_nl) == len(y_train_nl) and len(X_train_hi) == len(y_train_hi) and len(X_dev_it) == len(y_dev_it) and len(X_dev_pt) == len(y_dev_pt)
    print("PREDICTIONS FOR {0}".format(args.data_file))
    print("predictions for nl")
    make_predictions_nl(clf_params, X_train_nl, y_train_nl, X_dev_nl, y_dev_nl, stacked=True)
    print("predictions for de")
    make_predictions_nl(clf_params, X_train_de, y_train_de, X_dev_de, y_dev_de, stacked=True)
    print("predictions for es")
    make_predictions_nl(clf_params, X_train_es, y_train_es, X_dev_es, y_dev_es, stacked=True)
    print("predictions for it")
    make_predictions_nl(clf_params, X_train_it, y_train_it, X_dev_it, y_dev_it, stacked=True)
    print("predictions for pt")
    make_predictions_nl(clf_params, X_train_pt, y_train_pt, X_dev_pt, y_dev_pt, stacked=True)
    print("predictions for ru")
    make_predictions_nl(clf_params, X_train_ru, y_train_ru, X_dev_ru, y_dev_ru, stacked=True)
    print("predictions for po")
    make_predictions_nl(clf_params, X_train_po, y_train_po, X_dev_po, y_dev_po, stacked=True)
    print("predictions for pr")
    make_predictions_nl(clf_params, X_train_pr, y_train_pr, X_dev_pr, y_dev_pr, stacked=True)
    print("predictions for hi")
    make_predictions_nl(clf_params, X_train_hi, y_train_hi, X_dev_hi, y_dev_hi, stacked=True)
    #predictions mixed with same size as X_train_male or X_train_female (shuffled)
    X_train_mixed = X_train_nl[:round(len(X_train_nl)*(1/9))] + X_train_de[:round(len(X_train_de)*(1/9))] + X_train_es[:round(len(X_train_es)*(1/9))] + X_train_it[:round(len(X_train_it)*(1/9))] + X_train_pt[:round(len(X_train_pt)*(1/9))] + X_train_ru[:round(len(X_train_ru)*(1/9))] + X_train_po[:round(len(X_train_po)*(1/9))] + X_train_hi[:round(len(X_train_hi)*(1/9))] + X_train_pr[:round(len(X_train_pr)*(1/9))]
    y_train_mixed = y_train_nl[:round(len(y_train_nl)*(1/9))] + y_train_de[:round(len(y_train_de)*(1/9))] +  y_train_es[:round(len(y_train_es)*(1/9))] +  y_train_it[:round(len(y_train_it)*(1/9))] +  y_train_pt[:round(len(y_train_pt)*(1/9))] +  y_train_ru[:round(len(y_train_ru)*(1/9))] +  y_train_po[:round(len(y_train_po)*(1/9))] +  y_train_hi[:round(len(y_train_hi)*(1/9))] +  y_train_pr[:round(len(y_train_pr)*(1/9))]
    X_dev_mixed = X_dev_nl[:round(len(X_dev_nl)*(1/9))] + X_dev_de[:round(len(X_dev_de)*(1/9))] + X_dev_es[:round(len(X_dev_es)*(1/9))] + X_dev_it[:round(len(X_dev_it)*(1/9))] + X_dev_pt[:round(len(X_dev_pt)*(1/9))] + X_dev_ru[:round(len(X_dev_ru)*(1/9))] + X_dev_po[:round(len(X_dev_po)*(1/9))] + X_dev_hi[:round(len(X_dev_hi)*(1/9))] + X_dev_pr[:round(len(X_dev_pr)*(1/9))]
    y_dev_mixed = y_dev_nl[:round(len(y_dev_nl)*(1/9))] + y_dev_de[:round(len(y_dev_de)*(1/9))] + y_dev_es[:round(len(y_dev_es)*(1/9))] + y_dev_it[:round(len(y_dev_it)*(1/9))] + y_dev_pt[:round(len(y_dev_pt)*(1/9))] + y_dev_ru[:round(len(y_dev_ru)*(1/9))] + y_dev_po[:round(len(y_dev_po)*(1/9))] + y_dev_hi[:round(len(y_dev_hi)*(1/9))] + y_dev_pr[:round(len(y_dev_pr)*(1/9))] 
    y_extra_train_mixed = ["nl" for item in range(round(len(y_train_nl)*(1/9)))] + ["de" for item in range(round(len(y_train_de)*(1/9)))] + ["es" for item in range(round(len(y_train_es)*(1/9)))] + ["it" for item in range(round(len(y_train_it)*(1/9)))] + ["pt" for item in range(round(len(y_train_pt)*(1/9)))] + ["ru" for item in range(round(len(y_train_ru)*(1/9)))] + ["po" for item in range(round(len(y_train_po)*(1/9)))] + ["hi" for item in range(round(len(y_train_hi)*(1/9)))] + ["pr" for item in range(round(len(y_train_pr)*(1/9)))]  
    y_extra_dev_mixed = ["nl" for item in range(round(len(y_dev_nl)*(1/9)))] + ["de" for item in range(round(len(y_dev_de)*(1/9)))] + ["es" for item in range(round(len(y_dev_es)*(1/9)))] + ["it" for item in range(round(len(y_dev_it)*(1/9)))] + ["pt" for item in range(round(len(y_dev_pt)*(1/9)))] + ["ru" for item in range(round(len(y_dev_ru)*(1/9)))] + ["po" for item in range(round(len(y_dev_po)*(1/9)))] + ["hi" for item in range(round(len(y_dev_hi)*(1/9)))] + ["pr" for item in range(round(len(y_dev_pr)*(1/9)))]
    assert len(y_train_mixed) == len(y_extra_train_mixed) and len(y_dev_mixed) == len(y_extra_dev_mixed)
    print(X_train_mixed[0], y_train_mixed[0], y_extra_train_mixed[0])
    #instance_one = X_train_mixed[0]
    X_train_mixed, y_train_mixed, y_extra_train_mixed = shuffle(X_train_mixed, y_train_mixed, y_extra_train_mixed, random_state=42)
    X_dev_mixed, y_dev_mixed = shuffle(X_dev_mixed, y_dev_mixed, random_state=42)
    #index_instance_one = X_train_mixed.index(instance_one)
    #print("Index ", index_instance_one)
    #print(X_train_mixed[index_instance_one], y_train_mixed[index_instance_one], y_extra_train_mixed[index_instance_one]) 
    print("predictions mixed") 
    make_predictions_nl(clf_params, X_train_mixed, y_train_mixed, X_dev_mixed, y_dev_mixed, stacked=True)

    print("predictions entire dataset with NL as additional feature")
    # predictions entire dataset with gender supplied as additional feature
    make_predictions_nl(clf_params, X_train, y_train, X_dev, y_dev, list(y_extra_train), list(y_extra_dev), stacked=True)

    print("predictions smaller mixed dataset with NL as additional feature")
    # predictions smaller mixed dataset (same dataset as in #3) with gender supplied as feature (shuffled)
    make_predictions_nl(clf_params,X_train_mixed, y_train_mixed, X_dev_mixed, y_dev_mixed,list(y_extra_train_mixed),list(y_extra_dev_mixed), stacked=True)

def get_equal_lists_gender(X_train_male, X_train_female, y_train_male, y_train_female):
    if len(X_train_male) < len(X_train_female):
        X_train_female = X_train_female[:len(X_train_male)]
        y_train_female = y_train_female[:len(y_train_male)]
    elif len(X_train_female) < len(X_train_male):
        X_train_male = X_train_male[:len(X_train_female)]
        y_train_male = y_train_male[:len(y_train_female)]

    return X_train_male, X_train_female, y_train_male, y_train_female

def predict_native_lang(clf_params, X_train, y_train, X_dev, y_dev, y_extra_train, y_extra_dev, downsized=False):
    ## in this case NL is first followed by GENDER in the csv
    ## questions: crossval or dev set?, filter X_dev and y_dev by gender as well or leave as is?

    X_train_male = [X_train[ix] for ix, item in enumerate(y_extra_train) if item == "male"]
    X_train_female = [X_train[ix] for ix, item in enumerate(y_extra_train) if item == "female"]
    y_train_male = [y_train[ix] for ix, item in enumerate(y_extra_train) if item == "male"]
    y_train_female = [y_train[ix] for ix, item in enumerate(y_extra_train) if item == "female"]
    X_dev_male = [X_dev[ix] for ix, item in enumerate(y_extra_dev) if item == "male"]
    X_dev_female = [X_dev[ix] for ix, item in enumerate(y_extra_dev) if item == "female"]
    y_dev_male = [y_dev[ix] for ix, item in enumerate(y_extra_dev) if item == "male"]
    y_dev_female = [y_dev[ix] for ix, item in enumerate(y_extra_dev) if item == "female"]

    if downsized:
        print("lengths before")
        print(len(X_train_male), len(X_train_female), len(X_dev_male), len(X_dev_female))
        print(len(X_dev_male))
        X_train_male, X_train_female, y_train_male, y_train_female = get_equal_lists_gender(X_train_male, X_train_female, y_train_male, y_train_female)
        X_dev_male, X_dev_female, y_dev_male, y_dev_female = get_equal_lists_gender(X_dev_male, X_dev_female, y_dev_male, y_dev_female)


        print("lengths after")
        print(len(X_train_male), len(X_train_female), len(X_dev_male), len(X_dev_female))
        print(len(X_dev_male))

        print(X_train[0])

    assert len(X_train_male) == len(y_train_male) and len(X_train_female) == len(y_train_female) and len(X_dev_male) == len(y_dev_male) and len(X_dev_female) == len(y_dev_female)
    print("PREDICTIONS FOR {0}".format(args.data_file))
    #predictions for males
    print("predictions for males")
    make_predictions_nl(clf_params, X_train_male, y_train_male, X_dev_male, y_dev_male, stacked=True)
    #predictions for females
    print("predictions for females")
    make_predictions_nl(clf_params, X_train_female, y_train_female, X_dev_female, y_dev_female, stacked=True)
    #predictions mixed with same size as X_train_male or X_train_female (shuffled)
    X_train_mixed = X_train_male[:round(len(X_train_male)*0.5)] + X_train_female[:round(len(X_train_female)*0.5)]
    y_train_mixed = y_train_male[:round(len(y_train_male)*0.5)] + y_train_female[:round(len(y_train_female)*0.5)]
    X_dev_mixed = X_dev_male[:round(len(X_dev_male)*0.5)] + X_dev_female[:round(len(X_dev_female)*0.5)]
    y_dev_mixed = y_dev_male[:round(len(y_dev_male)*0.5)] + y_dev_female[:round(len(y_dev_female)*0.5)]
    y_extra_train_mixed = ["male" for item in range(round(len(y_train_male)*0.5))] + ["female" for item in range(round(len(y_train_female)*0.5))]
    y_extra_dev_mixed = ["male" for item in range(round(len(y_dev_male)*0.5))] + ["female" for item in range(round(len(y_dev_female)*0.5))]
    assert len(y_train_mixed) == len(y_extra_train_mixed) and len(y_dev_mixed) == len(y_extra_dev_mixed)
    #print(X_train_mixed[0], y_train_mixed[0], y_extra_train_mixed[0])
    #instance_one = X_train_mixed[0]
    X_train_mixed, y_train_mixed, y_extra_train_mixed = shuffle(X_train_mixed, y_train_mixed, y_extra_train_mixed, random_state=42)
    X_dev_mixed, y_dev_mixed = shuffle(X_dev_mixed, y_dev_mixed, random_state=42)
    #index_instance_one = X_train_mixed.index(instance_one)
    #print("Index ", index_instance_one)
    print(len(y_extra_train_mixed), len(y_train_male), len(y_train_female))
    #print(X_train_mixed[index_instance_one], y_train_mixed[index_instance_one], y_extra_train_mixed[index_instance_one]) 
    print("predictions mixed") 
    make_predictions_nl(clf_params, X_train_mixed, y_train_mixed, X_dev_mixed, y_dev_mixed, stacked=True)

    print("predictions entire dataset with gender as additional feature")
    # predictions entire dataset with gender supplied as additional feature
    make_predictions_nl(clf_params, X_train, y_train, X_dev, y_dev, list(y_extra_train), list(y_extra_dev), stacked=True)

    print("predictions smaller mixed dataset with gender as additional feature")
    # predictions smaller mixed dataset (same dataset as in #3) with gender supplied as feature (shuffled)
    make_predictions_nl(clf_params,X_train_mixed, y_train_mixed, X_dev_mixed, y_dev_mixed,list(y_extra_train_mixed),list(y_extra_dev_mixed), stacked=True)



if __name__ == "__main__":
    args = create_arg_parser()

    
    X_train, y_train, X_dev, y_dev, X_test, y_test, y_extra_train, y_extra_dev, y_extra_test = load_data(args.data_file, stacked=True)

    ## NOTE THAT STACKED=TRUE was added to load_data and make_predictions_nl
    clf_ngrams_params = []

    # best_model_per_task = {"cross-native-lang-tm": {"model1": {"ngram_range": (1,5), "analyzer": "char", "C": 1, "kernel": "linear"}},
    #                      "cross-native-lang-mt": {"model1": {"ngram_range": (1,5), "analyzer": "char", "C":10, "kernel": "linear"}},
    #                      "cross-gender-tm": {"model1": {"ngram_range": (1,2), "analyzer": "word", "C":10, "kernel": "linear"}},
    #                      "cross-gender-mt": {"model1": {"ngram_range": (1,1), "analyzer": "word", "C": 20, "kernel": "linear"}},
    #                      "within-native-lang-twitter": {"model1": {"ngram_range": (1,4), "analyzer": "char", "C": 20, "kernel": "linear"}},
    #                       "within-native-lang-medium": {"model1": {"ngram_range": (1,5), "analyzer": "char", "C": 10, "kernel": "linear"}},
    #                      "within-gender-twitter": {"model1": {"ngram_range": (1,4), "analyzer": "char", "C": 10, "kernel": "linear"}},
    #                      "within-gender-medium": {"model1": {"ngram_range": (1,7), "analyzer": "char", "C": 10, "kernel": "linear"}}}

    # clf_ngrams_params = get_best_model_for_task(best_model_per_task)

    # if args.cross and args.gender and args.medium == "twitter":
    #     # best features are POS/FUNCTION features
    #     X_train  = [nltk.word_tokenize(item) for item in X_train]
    #     X_dev = [nltk.word_tokenize(item) for item in X_dev]
    #     X_test = [nltk.word_tokenize(item) for item in X_test]
    #     X_train = [" ".join([item[1] if item[1].startswith("NN") or item[1].startswith("VB") else item[0] for item in nltk.pos_tag(text)]) for text in X_train]
    #     X_dev = [" ".join([item[1]  if item[1].startswith("NN") or item[1].startswith("VB") else item[0] for item in nltk.pos_tag(text) ]) for text in X_dev]
    #     X_test = [" ".join([item[1] if item[1].startswith("NN") or item[1].startswith("VB") else item[0] for item in nltk.pos_tag(text) ]) for text in X_test]
    #     print(X_train[0])

    if not args.gender:
        predict_native_lang(clf_ngrams_params, X_train, y_train, X_dev, np.array(y_dev), y_extra_train, np.array(y_extra_dev), downsized=True)
    else:
        #print(X_train[0], y_extra_train[0], y_train[0])
        predict_gender(clf_ngrams_params, X_train, y_train, X_dev, np.array(y_dev), y_extra_train, np.array(y_extra_dev), downsized=True)
        

    ## TO DO: stacked=True opruimen zodat het wat netter is, even meegeven als command line args