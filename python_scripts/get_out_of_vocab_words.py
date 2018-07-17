from nltk import corpus
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn import svm
from sklearn.pipeline import Pipeline, FeatureUnion
import numpy as np
import argparse
import pandas as pd 
from collections import Counter
from nltk.corpus import wordnet as wn
import itertools
import json

def create_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-infile","--infile", type=str, help="File with data")
    parser.add_argument("-best_model","--best_model", type=str, help="File with best ngram model for obtaining most inform. feats")
    parser.add_argument("-cross","--cross", action="store_true", help="Whether it is cross genre or not")
    parser.add_argument("-g","--gender", action="store_true", help="Whether the type is gender or not")
    parser.add_argument("-m","--medium", required=True, type=str, help="Which medium if within genre")
    args = parser.parse_args()
    return args

def load_data_regular(data_file, load_data=True, header_present=[0]):
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

def get_best_models_for_task(model_task_dict, model="model2"):
    if args.cross:
        if not args.gender:
            print(args.gender)
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
        if not args.gender:
            print(args.gender)
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


def load_data(data_file, load_data=True, header_present=[0]):
    datafile = pd.read_csv(data_file, header=header_present)
    dutch = [item[0] for item in datafile.loc[datafile['label'] == "the-netherlands"].iloc[:,:-3].values]
    german = [item[0] for item in datafile.loc[datafile['label'] == "germany"].iloc[:,:-3].values]
    spanish = [item[0] for item in datafile.loc[datafile['label'] == "spain"].iloc[:,:-3].values]
    portuguese = [item[0] for item in datafile.loc[datafile['label'] == "portugal"].iloc[:,:-3].values]
    italian = [item[0] for item in datafile.loc[datafile['label'] == "italy"].iloc[:,:-3].values]
    polish = [item[0] for item in datafile.loc[datafile['label'] == "poland"].iloc[:,:-3].values]
    russian = [item[0] for item in datafile.loc[datafile['label'] == "russia"].iloc[:,:-3].values]
    persian = [item[0] for item in datafile.loc[datafile['label'] == "iran"].iloc[:,:-3].values]
    hindi = [item[0] for item in datafile.loc[datafile['label'] == "new-delhi"].iloc[:,:-3].values]

    female = [item[0] for item in datafile.loc[datafile['second_label'] == "female"].iloc[:,:-3].values]
    male = [item[0] for item in datafile.loc[datafile['second_label'] == "male"].iloc[:,:-3].values]
    assert len(dutch) > 1 and len(german) > 1 and len(spanish) > 1 and len(portuguese) > 1 and len(italian) > 1 and len(polish) > 1 and len(russian) > 1 and len(persian) > 1 and len(hindi) > 1
    
    return dutch, german, spanish, portuguese, italian, polish, russian, persian, hindi, female, male

def unusual_words(texts):
    english_vocab = set(w.lower() for w in corpus.words.words())
    unusual_total = Counter()
    for text in texts:
        text_vocab = set(w.lower() for w in text.split() if w.isalpha())
        unusual = text_vocab - english_vocab
        for word in unusual:
            if not wn.synsets(word):
                unusual_total[word] += 1
            else:
                pass
                #print(wn.synsets(word))

    return unusual_total

def n_split(iterable, n, fillvalue=None):
        num_extra = len(iterable) % n
        zipped = zip(*[iter(iterable)] * n)
        print(list(zipped))
        return list(zipped) if not num_extra else list(zipped) + [list(iterable)[-num_extra:], ]

def create_retrofitting_file(*nargs):
    with open("retrofitting_file_oov_native_lang.txt", "w", encoding="utf-8") as outfile:
        for item in nargs:
            for group in n_split(item, 5):
                combinations = list(itertools.permutations(list(group), len(group)))
                for combination in combinations:
                    outfile.write(" ".join(list(combination)))
    print("done writing to file")

def print_top10(vectorizer, clf, class_labels):
    """Prints features with the highest coefficient values, per class"""
    feature_names = vectorizer.get_feature_names()
    #print(feature_names)
    svm_coef = clf.coef_
    with open("retrofitted_most_informative_feats_nativel.txt", "a", encoding="utf-8") as outfile:
        for i, class_label in enumerate(class_labels):
            top10 = sorted(zip(svm_coef[0], feature_names))[-5:]
            
            features = []
            print(class_label)
            for coef, feat in top10:
                print(coef, feat)
                features.append(feat)
            # combos = list(itertools.permutations(features, 5))
            # for combo in combos:
            #     outfile.write(" ".join(list(combo)))
            #     outfile.write("\n")

def get_most_informative_feats(pipeline, X_train, y_train, X_test, y_test):
    pipeline.fit(X_train, y_train)
    preds = pipeline.predict(X_test)
    print(classification_report(y_test, preds))
    print_top10(pipeline.named_steps["vect"], pipeline.named_steps["clf"], pipeline.classes_)

if __name__ == "__main__":
    args = create_arg_parser()
    # load data cross genre per country
    # method: filter words whcih don't occur in word list and wordnet, then take items which have freq of > 1, compare to other country lists and voila
    dutch, german, spanish, portuguese, italian, polish, russian, persian, hindi, female, male = load_data(args.infile)
    dutch_oov = [word for word, freq in unusual_words(dutch).items() if freq > 1]
    german_oov = [word for word, freq in unusual_words(german).items() if freq > 1]
    spanish_oov =[word for word, freq in  unusual_words(spanish).items() if freq > 1]
    portuguese_oov = [word for word, freq in unusual_words(portuguese).items() if freq > 1]
    italian_oov = [word for word, freq in unusual_words(italian).items() if freq > 1]
    polish_oov = [word for word, freq in unusual_words(polish).items() if freq > 1]
    russian_oov = [word for word, freq in unusual_words(russian).items() if freq > 1]
    persian_oov = [word for word, freq in unusual_words(persian).items() if freq > 1]
    hindi_oov = [word for word, freq in unusual_words(hindi).items() if freq > 1]
    #print(dutch_oov)
    dutch_filtered = set(dutch_oov) - (set(german_oov) | set(spanish_oov) | set(portuguese_oov) | set(italian_oov) | set(polish_oov) | set(russian_oov) | set(persian_oov) | set(hindi_oov))
    german_filtered = set(german_oov) - (set(dutch_oov) | set(spanish_oov) | set(portuguese_oov) | set(italian_oov) | set(polish_oov) | set(russian_oov) | set(persian_oov) | set(hindi_oov))
    spanish_filtered = set(spanish_oov) - (set(dutch_oov) | set(german_oov) | set(portuguese_oov) | set(italian_oov) | set(polish_oov) | set(russian_oov) | set(persian_oov) | set(hindi_oov))
    portuguese_filtered = set(portuguese_oov) -  (set(dutch_oov) | set(german_oov) | set(spanish_oov) | set(italian_oov) | set(polish_oov) | set(russian_oov) | set(persian_oov) | set(hindi_oov))
    italian_filtered = set(italian_oov) - (set(dutch_oov) | set(german_oov) | set(spanish_oov) | set(portuguese_oov) | set(polish_oov) | set(russian_oov) | set(persian_oov) | set(hindi_oov))
    polish_filtered = set(polish_oov) - (set(dutch_oov) | set(german_oov) | set(spanish_oov) | set(portuguese_oov) | set(italian_oov) | set(russian_oov) | set(persian_oov) | set(hindi_oov))
    russian_filtered = set(russian_oov) - (set(dutch_oov) | set(german_oov) | set(spanish_oov) | set(portuguese_oov) | set(italian_oov) | set(polish_oov) | set(persian_oov) | set(hindi_oov))
    persian_filtered = set(persian_oov) - (set(dutch_oov) | set(german_oov) | set(spanish_oov) | set(portuguese_oov) | set(italian_oov) | set(polish_oov) | set(russian_oov) | set(hindi_oov))
    hindi_filtered = set(hindi_oov) - (set(dutch_oov) | set(german_oov) | set(spanish_oov) | set(portuguese_oov) | set(italian_oov) | set(polish_oov) | set(russian_oov) | set(persian_oov))
    print(portuguese_filtered)
    create_retrofitting_file(dutch_filtered, german_filtered, spanish_filtered, portuguese_filtered, italian_filtered, polish_filtered, russian_filtered, persian_filtered, hindi_filtered)

    # female_oov = [word for word, freq in unusual_words(female).items() if freq > 1]
    # male_oov = [word for word, freq in unusual_words(male).items() if freq > 1]
    # female_filtered = set(female_oov) - set(male_oov)
    # male_filtered = set(male_oov) - set(female_oov)
    # print(female_filtered)
    # print(male_filtered)
    ## too long these lists > if time > filter and run
    #create_retrofitting_file(female_filtered, male_filtered)


    # X_train, y_train, X_dev, y_dev, X_test, y_test = load_data_regular(args.infile)
    # with open("best_regular_word_ngram_models.json".format(args.best_model), "r") as fp:
    #     best_models_per_task = json.load(fp)

    # best_model = get_best_models_for_task(best_models_per_task)

    
    # vect = TfidfVectorizer(ngram_range=best_model["ngram_range"], analyzer=best_model["analyzer"])
    # clf = svm.LinearSVC(C=best_model["C"])
    # pipeline = Pipeline([("vect", vect), ("clf", clf)])
    # pipeline.fit(X_train, y_train)
    # get_most_informative_feats(pipeline, X_train, y_train, X_dev, y_dev)
    
    ## get most informative features for within genre for both twitter and medium and combine (set) and select
    ## do the same for gender