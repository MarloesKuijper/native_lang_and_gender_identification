### cross val, mixed model, best SVM models predict proba
import argparse
import pandas as pd
import numpy as np 
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
import nltk

def create_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-name","--name", type=str, help="Name")
    parser.add_argument("-data_file","--data_file", type=str, help="Datafile")
    parser.add_argument("-crossval", "--crossval", action="store_true", help="If cross val has to be performed")
    parser.add_argument("-stackedfile","--stackedfile", type=str, help="Stacked features")
    parser.add_argument("-twitter_file","--twitter_file", type=str, help="Twitter datafile NL - G")
    parser.add_argument("-medium_file","--medium_file", type=str, help="Medium datafile NL - G")
    parser.add_argument("-stacked", "--stacked", action="store_true", help="If stacked features are used")
    parser.add_argument("-posfunc", "--posfunc", action="store_true", help="If pos + function words are used")
    parser.add_argument("-ngrams", "--ngrams", nargs="+", type=int, help="Which ngram range to use")
    parser.add_argument("-analyzer", "--analyzer", type=str, help="Which analyzer to use")
    parser.add_argument("-C", "--C", type=int, help="Which C value to use")
    parser.add_argument("-best_models", "--best_models", action="store_true", help="If you want to run best models or do cross-val/mixed models instead")
    args = parser.parse_args()
    return args


def load_data(data_file, load_extra_data=True, header_present=[0]):
    datafile = pd.read_csv(data_file, header=header_present)
    train = datafile.loc[datafile.iloc[:,-1] == "train"]
    test = datafile.loc[datafile.iloc[:,-1] == "test"]
    dev = datafile.loc[datafile.iloc[:,-1] == "dev"]
    if header_present:
        X_train = [item[0] for item in train.iloc[:,:-3].values]
        X_dev = [item[0] for item in dev.iloc[:,:-3].values]
        X_test = [item[0] for item in test.iloc[:,:-3].values]
    else:
        X_train = [item for item in train.iloc[:,:-3].values]
        X_dev = [item for item in dev.iloc[:,:-3].values]
        X_test = [item for item in test.iloc[:,:-3].values]

    y_train = train.iloc[:,-3]
    y_dev = dev.iloc[:,-3]
    y_test = test.iloc[:,-3]
    if load_extra_data:
        y_train_g = train.iloc[:,-2]
        y_dev_g = dev.iloc[:,-2]
        y_test_g = test.iloc[:,-2]
        
        return X_train, y_train, X_dev, y_dev, X_test, y_test, y_train_g, y_dev_g, y_test_g
    else:
        return X_train, y_train, X_dev, y_dev, X_test, y_test

def load_for_mixed_model(data_file, load_data=True, header_present=[0]):
    datafile = pd.read_csv(data_file, header=header_present)
    X = [item[0] for item in datafile.iloc[:,:-3].values]
    y = datafile.iloc[:,-3]
    y_extra = datafile.iloc[:,-2]
    
    return X,y, y_extra

def cross_val(X_train, y_train, X_dev, y_dev, X_test, y_test):
    ## hypothesized to be better than cross-genre but worse than within?
    ## kan eventueel 6 keer: 1x cross voor G en NL, 2x within voor G en NL (voor T en M afzonderlijk)
    X = X_train + X_dev + X_test
    y = list(y_train) + list(y_dev) + list(y_test)
    clf = SVC(C=1, kernel="linear")
    vect = TfidfVectorizer(ngram_range=(1,1), analyzer="word")
    pipeline = Pipeline([("vect", vect),("clf", clf)])
    print(cross_val_score(pipeline, X, y))

def mixed_model(X_t, y_t_nl, y_t_g, X_m, y_m_nl, y_m_g):
    y_t_nl = list(y_t_nl)
    y_t_g = list(y_t_g)
    y_m_nl = list(y_m_nl)
    y_m_g = list(y_m_g)
    X_train = X_t[:round(len(X_t)*(4/6))] + X_m[:round(len(X_m)*(4/6))]
    X_dev_twitter = X_t[round(len(X_t)*(4/6)):round(len(X_t)*(5/6))] 
    X_test_twitter = X_t[round(len(X_t)*(5/6)):round(len(X_t)*(6/6))]
    X_dev_medium = X_m[round(len(X_m)*(4/6)):round(len(X_m)*(5/6))]
    X_test_medium = X_m[round(len(X_m)*(5/6)):round(len(X_m)*(6/6))]
    y_train_nl = y_t_nl[:round(len(y_t_nl)*(4/6))] + y_m_nl[:round(len(y_m_nl)*(4/6))]
    y_dev_twitter_nl = y_t_nl[round(len(y_t_nl)*(4/6)):round(len(y_t_nl)*(5/6))] 
    y_test_twitter_nl = y_t_nl[round(len(y_t_nl)*(5/6)):round(len(y_t_nl)*(6/6))] 
    y_dev_medium_nl = y_m_nl[round(len(y_m_nl)*(4/6)):round(len(y_m_nl)*(5/6))]
    y_test_medium_nl = y_m_nl[round(len(y_m_nl)*(5/6)):round(len(y_m_nl)*(6/6))]
    y_train_g = y_t_g[:round(len(y_t_g)*(4/6))] + y_m_g[:round(len(y_m_g)*(4/6))]
    y_dev_twitter_g = y_t_g[round(len(y_t_g)*(4/6)):round(len(y_t_g)*(5/6))] 
    y_test_twitter_g =  y_t_g[round(len(y_t_g)*(5/6)):round(len(y_t_g)*(6/6))] 
    y_dev_medium_g = y_m_g[round(len(y_m_g)*(4/6)):round(len(y_m_g)*(5/6))]
    y_test_medium_g = y_m_g[round(len(y_m_g)*(5/6)):round(len(y_m_g)*(6/6))]
    ## missing the final item ??
    print(len(X_train), len(X_dev_twitter), len(X_test_twitter), len(X_dev_medium), len(X_test_medium))

    assert len(X_dev_twitter) == len(X_test_twitter) and len(X_dev_medium) == len(X_test_medium)

    clf = SVC(C=1, kernel="linear")
    vect = TfidfVectorizer(ngram_range=(1,1), analyzer="word")
    pipeline_nl = Pipeline([("vect", vect),("clf", clf)])
    pipeline_nl.fit(X_train, y_train_nl)
    preds_twitter_nl = pipeline_nl.predict(X_dev_twitter)
    print("Accuracy testing on twitter for NL")
    print(accuracy_score(y_dev_twitter_nl, preds_twitter_nl))
    preds_medium_nl = pipeline_nl.predict(X_dev_medium)
    print("Accuracy testing on medium for NL")
    print(accuracy_score(y_dev_medium_nl, preds_medium_nl))

    pipeline_g = Pipeline([("vect", vect), ("clf", clf)])
    pipeline_g.fit(X_train, y_train_g)
    preds_twitter_gender = pipeline_g.predict(X_dev_twitter)
    print("Accuracy testing on twitter for Gender")
    print(accuracy_score(y_dev_twitter_g, preds_twitter_gender))
    preds_medium_gender = pipeline_g.predict(X_dev_medium)
    print("Accuracy testing on medium for Gender")
    print(accuracy_score(y_dev_medium_g, preds_medium_gender))


def run_best_svm_models(X_train, y_train, X_dev, y_dev, X_test, y_test):
    if not args.stacked:
        clf = SVC(C=args.C, kernel="linear", random_state=42)
        vect = TfidfVectorizer(ngram_range=list(args.ngrams), analyzer=args.analyzer)
        pipeline = Pipeline([("vect", vect), ("clf", clf)])
        pipeline.fit(X_train, y_train)
        dev_preds = pipeline.predict(X_dev)
        dev_acc = accuracy_score(y_dev, dev_preds)
        test_preds = pipeline.predict(X_test)
        test_acc = accuracy_score(y_test, test_preds)
        print("Dev acc", dev_acc)
        print("Test acc", test_acc)


        clf_prob = SVC(C=args.C, kernel="linear", random_state=42, probability=True)
        pipeline_prob = Pipeline([("vect", vect), ("clf", clf_prob)])
        pipeline_prob.fit(X_train, y_train)
        dev_probs = pipeline_prob.predict_proba(X_dev)
        test_probs = pipeline_prob.predict_proba(X_test)
        np.save("dev_probs_svm_{0}.npy".format(args.name), dev_probs)
        np.save("test_probs_svm_{0}.npy".format(args.name), test_probs)
    else:
        # X_train = np.array(X_train)
        # print(X_train[:5])
        # X_dev = np.array(X_dev).reshape(-1,1)
        # X_test = np.array(X_test).reshape(-1,1)
        clf = SVC(C=1, kernel="linear", random_state=42)
        clf.fit(X_train, y_train)
        dev_preds = clf.predict(X_dev)
        test_preds = clf.predict(X_test)
        dev_acc = accuracy_score(y_dev, dev_preds)
        test_acc = accuracy_score(y_test, test_preds)
        print("Dev acc", dev_acc)
        print("Test acc", test_acc)

        clf_prob = SVC(C=1, kernel="linear", random_state=42, probability=True)
        clf_prob.fit(X_train, y_train)

        dev_probs = clf_prob.predict_proba(X_dev)
        test_probs = clf_prob.predict_proba(X_test)
        
        np.save("dev_probs_svm_{0}.npy".format(args.name), dev_probs)
        np.save("test_probs_svm_{0}.npy".format(args.name), test_probs)



if __name__ == "__main__":
    args = create_arg_parser()
    if not args.best_models:
        if args.crossval:
            X_train, y_train, X_dev, y_dev, X_test, y_test, y_train_g, y_dev_g, y_test_g = load_data(args.data_file)
            X_train = [" ".join(nltk.word_tokenize(item)) for item in X_train]
            X_dev = [" ".join(nltk.word_tokenize(item)) for item in X_dev]
            X_test = [" ".join(nltk.word_tokenize(item)) for item in X_test]
            print("Cross val for Native Language..")
            cross_val(X_train, y_train, X_dev, y_dev, X_test, y_test)
            print("Cross val for Gender..")
            cross_val(X_train, y_train_g, X_dev, y_dev_g, X_test, y_test_g)
        else:

            X_twitter, y_twitter, y_extra_twitter = load_for_mixed_model(args.twitter_file)
            X_medium, y_medium, y_extra_medium = load_for_mixed_model(args.medium_file)

            min_length = min([len(X_twitter), len(X_medium)])
            print("minimum length ", min_length)
            X_twitter, y_twitter, y_extra_twitter, X_medium, y_medium, y_extra_medium = X_twitter[:min_length], y_twitter[:min_length], y_extra_twitter[:min_length], X_medium[:min_length], y_medium[:min_length], y_extra_medium[:min_length]
            print("Lengths: ", len(X_twitter), len(y_twitter), len(y_extra_twitter), len(X_medium), len(y_medium), len(y_extra_medium))
            print("Running mixed models")
            mixed_model(X_twitter, y_twitter, y_extra_twitter, X_medium, y_medium, y_extra_medium)
            ## cross val: 2 versions: gender and native lang
            ## mixed model: 4 versions: test on twitter > G, NL, test on medium > G, NL
            ## best models: 8 (1 per task)

    else:
        if args.stacked:
            X_train, y_train, X_dev, y_dev, X_test, y_test = load_data(args.stackedfile, load_extra_data=False, header_present=None)
        else:
            X_train, y_train, X_dev, y_dev, X_test, y_test = load_data(args.data_file, load_extra_data=False)
        if args.posfunc:
            X_train  = [nltk.word_tokenize(item) for item in X_train]
            X_dev = [nltk.word_tokenize(item) for item in X_dev]
            X_test = [nltk.word_tokenize(item) for item in X_test]
            X_train = [" ".join([item[1] if item[1].startswith("NN") or item[1].startswith("VB") else item[0] for item in nltk.pos_tag(text)]) for text in X_train]
            X_dev = [" ".join([item[1]  if item[1].startswith("NN") or item[1].startswith("VB") else item[0] for item in nltk.pos_tag(text) ]) for text in X_dev]
            X_test = [" ".join([item[1] if item[1].startswith("NN") or item[1].startswith("VB") else item[0] for item in nltk.pos_tag(text) ]) for text in X_test]
        print(args.ngrams)
        run_best_svm_models(X_train, y_train, X_dev, y_dev, X_test, y_test)

        # cross_native_lang_tm = ngram_range 1,5, analyzer char, kernel linear, c = 1
        # cross_native_lang_mt = ngram range 1,5 analyzer char, kernel linear, c= 10
        # cross_gender_tm = ngram range 1,2 analyzer word, kernel linear, c=1 posfunc nodig
        # cross_gender_mt = ngram range 1,1, analyzer word, kernel linear, c=  20
        # within_native_lang_twitter = NA (ling feat nodig!!)
        # within_native_lang_medium = ngram range 1,5, analyzer char, kernel linear, c=10
        # within_gender_twitter = NA (ling feat nodig!!)
        # within_gender_medium = ngram range 1,7, analyzer char, kernel linear, C=10
