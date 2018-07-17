import os, sys, re, subprocess, shlex, argparse
import numpy as np
from sklearn import svm
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score
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

#### sources word lists
# colours: https://www.vocabulary.com/lists/540059
# fillers:  https://github.com/words/fillers/blob/master/data.txt
# hedges: https://github.com/words/hedges/blob/master/data.txt
# swearwords:  https://raw.githubusercontent.com/words/profanities/master/support.md


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

def get_data_gender(path, countries, affix, excel_file):
    df = pd.read_excel(excel_file)

    folders = [name for name in os.listdir(path) if os.path.isdir(os.path.join(path, name)) and name in countries]
    dataset = defaultdict(dict)
    for folder in folders:
        #print(folder)
        contents = os.listdir(os.path.join(path,folder))
        for file in contents:
            if file.endswith(affix) or file.startswith(affix):
                if affix.endswith("@"):
                    username = "@" + file.split(affix)[-1][:-4]
                else:
                    username = file.split(affix)[0]
                try: ## temporary (still have to remove some names from folders)
                    index = list(df["Username"]).index(username)
                    gender = df["Gender"][index].strip()
                    #print(gender)
                    #print(gender.strip())
                    with open(os.path.join(path+folder+"/",file), "r", encoding="utf-8") as infile:
                        data = infile.read().split("\n\n")
                        for line in data:
                            if gender not in dataset[folder]:
                                dataset[folder][gender] = preprocess_text(line.strip(), affix, folder, pos_tagging=False)
                            else:
                                dataset[folder][gender] += " " + preprocess_text(line.strip(), affix, folder, pos_tagging=False)
                except ValueError:
                    pass

    #print(dataset["the-netherlands"]["female"])
    return dataset

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
    

def get_data(path, countries, affix):
    folders = [name for name in os.listdir(path) if os.path.isdir(os.path.join(path, name)) and name in countries]
    dataset = defaultdict(str)
    for folder in folders:
        contents = os.listdir(os.path.join(path,folder))
        for file in contents:
            if file.endswith(affix) or file.startswith(affix):
                with open(os.path.join(path+folder+"/",file), "r", encoding="utf-8") as infile:
                    data = infile.read().split("\n\n")
                    for line in data:
                        dataset[folder] += " " + preprocess_text(line.strip(), affix, folder, pos_tagging=False)

    #print(dataset["the-netherlands"])
    return dataset

def split_data(data, length):
    split_data = []
    list_data = data.split()
    sections = math.floor(len(list_data) / length)
    for i in range(sections):
        portion = list_data[i*length:(i*length)+length]
        # if i == 0:
        #     print(portion)
        split_data.append(" ".join(portion))

    #print(split_data[0])

    return split_data

def downsample_data(X,y):
    print("Len X before", len(X))
    counts = Counter(y)
    print(counts)
    max_portions = counts.most_common()[-1][1]
    print(max_portions)
    item_counts = dict()
    remove_indices = []
    for index, item in enumerate(y):
        #print(item)
        item_counts[item] = item_counts.get(item, 0) + 1

        if item_counts[item] > max_portions:
            remove_indices.append(index)

    #print(remove_indices)
    X = [item for ix, item in enumerate(X) if ix not in remove_indices]
    y = [item for ix, item in enumerate(y) if ix not in remove_indices]
    #print(item_counts)
    print(Counter(y))

    return X,y

def split_dataset_within(data, length, gender=False, downsample=False):
    X = []
    y = []
    if not gender:
        for k,v in data.items():
            portions = split_data(v, length)
            for portion in portions:
                X.append(portion)
                y.append(k)
    else:
        for country,v in data.items():
            for gndr,val in v.items():
                portions = split_data(val, length)
                for portion in portions:
                    X.append(portion)
                    y.append(gndr)

    if downsample:
        X, y = downsample_data(X,y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42, shuffle=True)

    return X_train, X_test, y_train, y_test

def split_dataset_cross(train, test, gender=False, downsample=False):
    X_train = []
    y_train = []
    X_test = []
    y_test = []

    if not gender:
        for k,v in train.items():
            portions = split_data(v, int(args.length))
            for portion in portions:
                X_train.append(portion)
                y_train.append(k)

        for k,v in test.items():
            portions = split_data(v, int(args.length))
            for portion in portions:
                X_test.append(portion)
                y_test.append(k)

    else:
        for country,v in train.items():
            for gndr,val in v.items():
                portions = split_data(val, int(args.length))
                for portion in portions:
                    X_train.append(portion)
                    y_train.append(gndr)
        for country,v in test.items():
            for gndr,val in v.items():
                portions = split_data(val, int(args.length))
                for portion in portions:
                    X_test.append(portion)
                    y_test.append(gndr)

    if downsample:
        X_train, y_train = downsample_data(X_train,y_train)
        X_test, y_test = downsample_data(X_test,y_test)

    return X_train, X_test, y_train, y_test


def do_ML_within_genre(data, length, gender=False, downsample=False,name=None):
    X_train, X_test, y_train, y_test = split_dataset_within(data, length, gender, downsample)
    X_test, y_test, X_dev, y_dev = X_test[:round(len(X_test)*0.5)], y_test[:round(len(y_test)*0.5)], X_test[round(len(X_test)*0.5):], y_test[round(len(y_test)*0.5):]

    with open("within_genre_gender_{0}_initial_test_run.txt".format(name), "w", encoding="utf-8") as outfile:
        for kernel in ["linear", "rbf", "poly"]:
            for c in [0.01,0.1,1,10]:
                vec = TfidfVectorizer() # add binary=True here
                clf = svm.SVC(kernel=kernel,C=c)
                clf = Pipeline( [('vec', vec), ('cls', clf)] )
                clf.fit(X_train, y_train)
                y_guess = clf.predict(X_dev)
                accuracy = accuracy_score(y_dev, y_guess)
                f1score = f1_score(y_dev, y_guess, average="macro")
                #cm = confusion_matrix(y_dev, y_guess)
                # print(accuracy, f1score)
                # print(classification_report(y_dev, y_guess))
                outfile.write("SVM with ngram range {0}, analyzer {1}, kernel {2}, C {3}\n".format("(1,1)", "word", kernel, c))
                outfile.write("Accuracy: {0}\n".format(accuracy))
                outfile.write("(Macro) F1-score: {0}\n".format(f1score))
                outfile.write(classification_report(y_dev, y_guess))
                outfile.write("\n\n")

def do_ML_cross_genre(train, test, gender=False, downsample=False):
    X_train, X_test, y_train, y_test = split_dataset_cross(train, test, gender, downsample)
    X_train, y_train = shuffle(X_train, y_train, random_state=42)
    X_test, y_test = shuffle(X_test, y_test, random_state=42)

    X_test, y_test, X_dev, y_dev = X_test[:round(len(X_test)*0.5)], y_test[:round(len(y_test)*0.5)], X_test[round(len(X_test)*0.5):], y_test[round(len(y_test)*0.5):]
    with open("cross_genre_gender_initial_testrun.txt", "w", encoding="utf") as outfile:
        for kernel in ["linear", "rbf", "poly"]:
            for c in [0.01, 0.1,1,10]:
                vec = TfidfVectorizer()
                clf = svm.SVC(kernel=kernel, C=c)
                clf = Pipeline( [('vec', vec), ('cls', clf)] )
                clf.fit(X_train, y_train)
                y_guess = clf.predict(X_dev)
                accuracy = accuracy_score(y_dev, y_guess)
                f1score = f1_score(y_dev, y_guess, average="macro")
                #cm = confusion_matrix(y_dev, y_guess)
                # print(accuracy, f1score)
                # print(classification_report(y_dev, y_guess))
                outfile.write("SVM with ngram range {0}, analyzer {1}, kernel {2}, C {3}\n".format("(1,1)", "word", kernel, c))
                outfile.write("Accuracy: {0}\n".format(accuracy))
                outfile.write("(Macro) F1-score: {0}\n".format(f1score))
                outfile.write(classification_report(y_dev, y_guess))
                outfile.write("\n\n")

def merge_datadicts_nl(dict1, dict2):
    ## dict2 is smallest
    dict3 = {}
    for k,v in dict1.items():
        dict3[k] = v
        if k in dict2:
            dict3[k] += dict2[k]

    return dict3

def merge_datadicts_gender(dict1, dict2):
    ## dict2 is smallest
    dict3 = defaultdict(dict)
    for country,v in dict1.items():
        for gender, val in v.items():
            dict3[country][gender] = val
            if gender in dict2[country]:
                dict3[country][gender] += dict2[country][gender]

    return dict3


def get_nl_data(countries):
    # get all data in dictionaries per country per genre
    twitter_data = get_data(args.path+"/twitter/", countries, "_messages.txt")
    # print("Twitter")
    # do_ML_within_genre(twitter_data, int(args.length))
    more_twitter_data = get_data(args.path+"/twitter_additional_data/", countries, "_messages.txt")
    full_twitter = merge_datadicts_nl(twitter_data, more_twitter_data)
    medium_data = get_data(args.path+"/medium/", countries, "messages_@")
    # print("Medium")
    # do_ML_within_genre(medium_data, int(args.length))
    more_medium_data = get_data(args.path+"/medium_additional_data/", countries, "messages_@")
    full_medium = merge_datadicts_nl(medium_data, more_medium_data)

    return twitter_data, full_twitter, medium_data, full_medium

def get_gender_data(countries):
    medium_data = get_data_gender(args.path+"/medium/", countries, "messages_@", "./results/META_DATA_CORPUS/data_medium_excel_filtered.xlsx")
    more_medium_data = get_data_gender(args.path+"/medium_additional_data/", countries, "messages_@", "./results/META_DATA_CORPUS/data_medium_excel_additional_data.xlsx")
    twitter_data = get_data_gender(args.path+"/twitter/", countries, "_messages.txt", "./results/META_DATA_CORPUS/data_twitter_excel_filtered.xlsx")
    more_twitter_data = get_data_gender(args.path+"/twitter_additional_data/", countries, "_messages.txt", "./results/META_DATA_CORPUS/data_twitter_excel_additional_data.xlsx")
    full_medium = merge_datadicts_gender(medium_data, more_medium_data)
    full_twitter = merge_datadicts_gender(twitter_data, more_twitter_data)

    return twitter_data, full_twitter, medium_data, full_medium

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


def get_multiple_negs(data_list):
    mult_negs = []
    for item in data_list:
        sentences = sent_tokenize(item)
        found = False
        for sentence in sentences:
            multiple_negatives = True if len([word for word in sentence.split() if word.lower() == "not" or word.lower() == "n't"]) > 1 else False
            if multiple_negatives == True:
                found = True
        if found:
            mult_negs.append(1)
        else:
            mult_negs.append(0)

    return mult_negs

def get_number_of_articles(data_list):
    avg_number_of_articles = []
    for item in data_list:
        sentences = sent_tokenize(item)
        articles = 0
        for sentence in sentences:
            tokens = word_tokenize(sentence)
            tagged = nltk.pos_tag(tokens)
            number_of_articles = np.sum([1 for word in tagged if word[1] == "DT"])
            articles += number_of_articles
        avg_articles = articles / len(sentences)
        avg_number_of_articles.append(avg_articles)

    return avg_number_of_articles

def get_capitalized_words(data_list):
    # https://liwc.wpengine.com/compare-dictionaries/
    capitalized_word_feats = []
    for item in data_list:
        sentences = sent_tokenize(item)
        capitalized_words = [word for item in sentences for word in item.split() if word[0].isupper()]
        capitalized_only = np.sum([1 for word in capitalized_words if len(word) > 2 and not word[1].isupper()]) / len(sentences)
        capitalized_word_feats.append(capitalized_only)

    return capitalized_word_feats

def get_avg_sent_length(data_list):
    # https://liwc.wpengine.com/compare-dictionaries/
    avg_sent_feats = []
    for item in data_list:
        sentences = sent_tokenize(item)
        avg_sent_length = np.mean([len(item.split()) for item in sentences])
        avg_sent_feats.append(avg_sent_length)
    return avg_sent_feats

def get_avg_adjectives(data_list):
    avg_no_adjectives = []
    for item in data_list:
        sentences = sent_tokenize(item)
        adjectives = 0
        for sentence in sentences:
            tokens = word_tokenize(sentence)
            tagged = nltk.pos_tag(tokens)
            no_adjectives = np.sum([1 for word in tagged if word[1] == "JJ"])
            adjectives += no_adjectives
        avg_adjectives = adjectives / len(sentences)
        avg_no_adjectives.append(avg_adjectives)
    return avg_no_adjectives

def get_sentiment(data_list):
    sentiments = []
    for item in data_list:
        sentences = sent_tokenize(item)
        sentiment_scores = []
        for sentence in sentences:
            sentiment = TextBlob(sentence)
            sentiment_scores.append(sentiment.sentiment.polarity)
        avg_sentiment = np.mean(sentiment_scores)
        sentiments.append(avg_sentiment)
    return sentiments

def get_colors(data_list):
    ## get one feature per instance for number of regular colour words and one feature for rare colour words
    # see list
    color_features = []
    with open("special_color_words.txt", "r", encoding="utf-8") as special_colors, open("regular_colors.txt", "r", encoding="utf-8") as regular_colors:
        special_c = [item.strip().lower() for item in special_colors.readlines()]
        regular_c = [item.strip().lower() for item in regular_colors.readlines()]
    for item in data_list:
        #tokens = word_tokenize(item)
        no_special_colors = 0
        no_regular_colors = 0
        for s_color in special_c:
            if s_color in item.lower():
                no_special_colors += 1
        for r_color in regular_c:
            if r_color in item.lower():
                no_regular_colors += 1
        color_features.append([no_special_colors, no_regular_colors])

    return color_features

def get_swearwords(data_list):
    # see list
    # MAYBE DO SIGMOID/SOFTMAX?
    with open("swearwords.txt", "r", encoding="utf-8") as swearword_file:
        swearwords = [item.strip().lower() for item in swearword_file.readlines()]
    swearword_features = []
    for item in data_list:
        #tokens = word_tokenize(item)
        no_swearwords = 0
        for swear in swearwords:
            if swear in item.lower():
                no_swearwords += 1
        swearword_features.append(no_swearwords)

    return swearword_features

def get_no_exclamation_and_question_marks(data_list):
    quest_excl_features = []
    for item in data_list:
        tokens = word_tokenize(item)
        no_question_marks = 0
        no_exclamation_marks = 0
        for token in tokens:
            if token  == "?":
                no_question_marks += 1
            elif token == "!":
                no_exclamation_marks += 1
        quest_excl_features.append([no_question_marks, no_exclamation_marks])

    return quest_excl_features

def get_hedges(data_list):
    # se list
    with open("hedges.txt", "r", encoding="utf-8") as hedge_file:
        hedges = [item.strip().lower() for item in hedge_file.readlines()]
    hedge_features = []
    for item in data_list:
        #tokens = word_tokenize(item)
        no_hedges = 0
        for hedge in hedges:
            if hedge in item.lower():
                no_hedges += 1
        hedge_features.append(no_hedges)

    return hedge_features

def get_fillers(data_list):
    # see list
    with open("fillers.txt", "r", encoding="utf-8") as fillers_file:
        fillers = [item.strip().lower() for item in fillers_file.readlines()]
    filler_features = []
    for item in data_list:
        tokens = word_tokenize(item)
        no_fillers = 0
        for token in tokens:
            if token in fillers:
                no_fillers += 1
        filler_features.append(no_fillers)

    return filler_features

def get_minimal_responses(data_list):
    minimal_responses = ["huh", "yeah", "mm", "mmm", "yep", "exactly", "hm", "hmm", "ok", "right"]
    minimal_response_features = []
    for item in data_list:
        tokens = word_tokenize(item)
        no_minimal_responses = len([item for item in tokens if item in minimal_responses])
        minimal_response_features.append(no_minimal_responses)

    return minimal_response_features

def get_word_order(instances, nlp):
    word_order_features = []
    for instance in instances:
        sentences_for_instance = sent_tokenize(instance)
        has_svo = 0
        has_sov = 0
        has_vos = 0
        has_vso = 0
        has_ovs = 0
        has_osv = 0
        for sentence in sentences_for_instance:
            doc = nlp(sentence)
            #print(doc)
            deps = []
            for token in doc:
                #print(token.text, token.dep_)
                deps.append((str(token.text),str(token.dep_)))
            deps_all = " ".join([dep if text != "," and text != ";" else "separator" for text, dep in deps ])
            deps_all = deps_all.split("separator")
            deps_all = [item.split() for item in deps_all]
            #print(deps_all)
            for item in deps_all:
                subj = None
                verb = None
                obj = None
                for dependency in item:
                    if dependency in ["nsubj", "nsubjpass"]:
                        #print("subj found")
                        subj = item.index(dependency)
                    elif dependency == "ROOT":
                        #print("verb found")
                        verb = item.index("ROOT")
                    elif dependency == "dobj":
                        #print("obj found")
                        obj = item.index("dobj")

                #print(subj, verb, obj)

                if type(subj) == int and type(verb) == int and type(obj) == int:
                    if subj < verb and subj < obj:
                        if verb < obj:
                            #print("word order is SVO")
                            has_svo += 1
                        else:
                            #print("word order is SOV")
                            has_sov += 1
                    elif obj < verb and obj < subj:
                        if verb < subj:
                            #print("word order is OVS")
                            has_ovs += 1
                        else:
                            #print("word order is OSV")
                            has_osv += 1
                    elif verb < obj and verb < subj:
                        if obj < subj:
                            #print("word order is VOS")
                            has_vos += 1
                        else:
                            #print("word order is VSO")
                            has_vso += 1
                else:
                    #print("sorry no order could be established")
                    pass
        word_order_features.append([has_svo, has_sov, has_vos, has_vso, has_ovs, has_osv])
    return word_order_features

def load_data(data_file, load_data=True, header_present=[0]):
    datafile = pd.read_csv(data_file, header=header_present)
    train = datafile.loc[datafile.iloc[:,-1] == "train"]
    test = datafile.loc[datafile.iloc[:,-1] == "test"]
    dev = datafile.loc[datafile.iloc[:,-1] == "dev"]
    if load_data:
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

    if load_data:
        y_extra_train = train.iloc[:,-2]
        y_extra_dev = dev.iloc[:,-2]
        y_extra_test = test.iloc[:,-2]
        return X_train, y_train, X_dev, y_dev, X_test, y_test, y_extra_train, y_extra_dev, y_extra_test
    else:
        return X_train, y_train, X_dev, y_dev, X_test, y_test


def combined_feats(*args):
    list_of_datasets = []
    for ix, data_list in enumerate(args):
        if ix not in [4,5,12]:
            data = np.array(data_list).reshape(-1,1)
        else:
            data = np.array(data_list)
        list_of_datasets.append(data)
    combined = np.hstack(list_of_datasets)
    return combined

def get_word_order_features(X_train, X_dev, X_test, y_train, y_dev, y_test, y_extra_train, y_extra_dev, y_extra_test):
    nlp = spacy.load('en_core_web_lg')
    X_train_features = np.array(get_word_order(X_train, nlp))
    X_dev_features = np.array(get_word_order(X_dev, nlp))
    X_test_features = np.array(get_word_order(X_test, nlp))
    train_dummy = np.array(["train" for item in y_train]).reshape(-1,1)
    dev_dummy = np.array(["dev" for item in y_dev]).reshape(-1,1)
    test_dummy = np.array(["test" for item in y_test]).reshape(-1,1)
    data_train = pd.DataFrame(np.hstack((X_train_features, np.array(y_train).reshape(-1,1), np.array(y_extra_train).reshape(-1,1), train_dummy)))
    data_dev = pd.DataFrame(np.hstack((X_dev_features, np.array(y_dev).reshape(-1,1), np.array(y_extra_dev).reshape(-1,1), dev_dummy)))
    data_test = pd.DataFrame(np.hstack((X_test_features, np.array(y_test).reshape(-1,1), np.array(y_extra_test).reshape(-1,1), test_dummy)))
    frames = [data_train, data_test, data_dev]
    data = pd.concat(frames)
    data.to_csv(args.ff, index=False, header=False)

def test_svm_params(train, test=None, gender=False, downsample=False, within=False, model_name=""):
    # if within:
    #     X_train, X_test, y_train, y_test = split_dataset_within(train, int(args.length), gender, downsample)
    # else:
    #     X_train, X_test, y_train, y_test = split_dataset_cross(train, test, gender, downsample)
    #     X_train, y_train = shuffle(X_train, y_train, random_state=42)
    #     X_test, y_test = shuffle(X_test, y_test, random_state=42)

    # X_test, y_test, X_dev, y_dev = X_test[:round(len(X_test)*0.5)], y_test[:round(len(y_test)*0.5)], X_test[round(len(X_test)*0.5):], y_test[round(len(y_test)*0.5):]
    # #keep this order in mind when using the pos-tagging features

    X_train, y_train, X_dev, y_dev, X_test, y_test, y_extra_train, y_extra_dev, y_extra_test = load_data(args.data_file, header_present=[0])
    #get_word_order_features(X_train, X_dev, X_test, y_train, y_dev, y_test, y_extra_train, y_extra_dev, y_extra_test)
    ## getwofeatures.sh

    # data_dict = {"avg_length_feats": {"X_train": get_avg_sent_length(X_train), "X_dev": get_avg_sent_length(X_dev), "X_test": get_avg_sent_length(X_test)},
    #             "capitalized_feats": {"X_train": get_capitalized_words(X_train), "X_dev": get_capitalized_words(X_dev), "X_test": get_capitalized_words(X_test)},
    #             "articles_feats": {"X_train": get_number_of_articles(X_train), "X_dev": get_number_of_articles(X_dev), "X_test": get_number_of_articles(X_test)},
    #             "mult_negs_feats": {"X_train": get_multiple_negs(X_train), "X_dev": get_multiple_negs(X_dev), "X_test": get_multiple_negs(X_test)},
    #             "combined_feats": {"X_train": combined_feats(get_avg_sent_length(X_train), get_capitalized_words(X_train), get_number_of_articles(X_train),get_multiple_negs(X_train)), "X_dev":
    #             combined_feats(get_avg_sent_length(X_dev), get_capitalized_words(X_dev), get_number_of_articles(X_dev), get_multiple_negs(X_dev)),
    #             "X_test": combined_feats(get_avg_sent_length(X_test), get_capitalized_words(X_test), get_number_of_articles(X_test), get_multiple_negs(X_test))}}          

    X_train_wo, y_train_wo, X_dev_wo, y_dev_wo, X_test_wo, y_test_wo = load_data(args.ff, load_data=False, header_present=None)

    print(args.data_file)
    print(np.array(y_train[:5]), np.array(y_train_wo[:5]))
    print()
    print(y_dev[:5], y_dev_wo[:5])
    print()
    print(y_test[:5], y_test_wo[:5])
    print()
    print()

    # length = get_avg_sent_length(X_train)
    # caps = get_capitalized_words(X_train)
    # articles = get_number_of_articles(X_train)
    # mult_negs = get_multiple_negs(X_train)
    # color = get_colors(X_train)
    # filler = get_fillers(X_train)
    # hedges = get_hedges(X_train)
    # swearwords = get_swearwords(X_train)
    # sentiment = get_sentiment(X_train)
    # minimal_responses = get_minimal_responses(X_train)
    # adjectives = get_avg_adjectives(X_train)
    # punct = get_no_exclamation_and_question_marks(X_train)
    # data = combined_feats(length, caps, articles, mult_negs, X_train_wo, color, filler, hedges, swearwords, sentiment, minimal_responses, adjectives, punct)

    # print("first row")
    # print(length[0], caps[0], articles[0], mult_negs[0], X_train_wo[0],color[0], filler[0], hedges[0], swearwords[0], sentiment[0], minimal_responses[0], adjectives[0], punct[0])
    # print(length[-1], caps[-1], articles[-1], mult_negs[-1], X_train_wo[-1],color[-1], filler[-1], hedges[-1], swearwords[-1], sentiment[-1], minimal_responses[-1], adjectives[-1], punct[-1])
    # print(data[0])
    # print(data[-1])

    # print("STOP")

    train_dummy = np.array(["train" for item in y_train]).reshape(-1,1)
    dev_dummy = np.array(["dev" for item in y_dev]).reshape(-1,1)
    test_dummy = np.array(["test" for item in y_test]).reshape(-1,1)


    data_dict = {"avg_length_feats": {"X_train": get_avg_sent_length(X_train), "X_dev": get_avg_sent_length(X_dev), "X_test": get_avg_sent_length(X_test)},
                "capitalized_feats": {"X_train": get_capitalized_words(X_train), "X_dev": get_capitalized_words(X_dev), "X_test": get_capitalized_words(X_test)},
                "articles_feats": {"X_train": get_number_of_articles(X_train), "X_dev": get_number_of_articles(X_dev), "X_test": get_number_of_articles(X_test)},
                "mult_negs_feats": {"X_train": get_multiple_negs(X_train), "X_dev": get_multiple_negs(X_dev), "X_test": get_multiple_negs(X_test)},
                  "word_order_feats": {"X_train": X_train_wo, "X_dev": X_dev_wo, "X_test": X_test_wo},
                 "color_feats": {"X_train": get_colors(X_train), "X_dev": get_colors(X_dev), "X_test": get_colors(X_test)},
                 "filler_feats": {"X_train": get_fillers(X_train), "X_dev": get_fillers(X_dev), "X_test": get_fillers(X_test)},
                 "hedge_feats": {"X_train": get_hedges(X_train), "X_dev": get_hedges(X_dev), "X_test": get_hedges(X_test)},
                 "swearword_feats": {"X_train": get_swearwords(X_train), "X_dev": get_swearwords(X_dev), "X_test": get_swearwords(X_test)},
                 "sentiment_feats": {"X_train": get_sentiment(X_train), "X_dev": get_sentiment(X_dev), "X_test": get_sentiment(X_test)},
                 "minimal_response_feats": {"X_train": get_minimal_responses(X_train), "X_dev": get_minimal_responses(X_dev), "X_test": get_minimal_responses(X_test)},
                 "adjective_feats": {"X_train": get_avg_adjectives(X_train), "X_dev": get_avg_adjectives(X_dev), "X_test": get_avg_adjectives(X_test)},
                 "punct_feats": {"X_train": get_no_exclamation_and_question_marks(X_train), "X_dev": get_no_exclamation_and_question_marks(X_dev), "X_test": get_no_exclamation_and_question_marks(X_test)},
                "combined_feats_all13": {"X_train": combined_feats(get_avg_sent_length(X_train), get_capitalized_words(X_train), get_number_of_articles(X_train),get_multiple_negs(X_train), X_train_wo,
                    get_colors(X_train), get_fillers(X_train), get_hedges(X_train), get_swearwords(X_train), get_sentiment(X_train), get_minimal_responses(X_train), get_avg_adjectives(X_train), get_no_exclamation_and_question_marks(X_train)),
                 "X_dev": combined_feats(get_avg_sent_length(X_dev), get_capitalized_words(X_dev), get_number_of_articles(X_dev), get_multiple_negs(X_dev), X_dev_wo,
                    get_colors(X_dev), get_fillers(X_dev), get_hedges(X_dev), get_swearwords(X_dev), get_sentiment(X_dev), get_minimal_responses(X_dev), get_avg_adjectives(X_dev), get_no_exclamation_and_question_marks(X_dev)),
                "X_test": combined_feats(get_avg_sent_length(X_test), get_capitalized_words(X_test), get_number_of_articles(X_test), get_multiple_negs(X_test), X_test_wo, get_colors(X_test), 
                    get_fillers(X_test),get_hedges(X_test), get_swearwords(X_test), get_sentiment(X_test), get_minimal_responses(X_test), get_avg_adjectives(X_test), get_no_exclamation_and_question_marks(X_test))}}

    # # word order 4, colors 5, excl 12

    # ##### TO DO: add additional label HEREE (DOWN )
    column_values = ["sent_length", "capitalization", "articles", "mult_negs", "has_svo", "has_sov", "has_vos", "has_vso", "has_ovs", "has_osv", "color_special", "color_regular", "filler", "hedges", "swearwords", "sentiment", "min_response", "adjectives", "punct_?", "punct_!","y_label", "extra_label", "set_type"]
    data_train = pd.DataFrame(np.hstack((data_dict["combined_feats_all13"]["X_train"], np.array(y_train).reshape(-1,1), np.array(y_extra_train).reshape(-1,1), train_dummy)), columns=column_values)
    data_dev = pd.DataFrame(np.hstack((data_dict["combined_feats_all13"]["X_dev"], np.array(y_dev).reshape(-1,1), np.array(y_extra_dev).reshape(-1,1),dev_dummy)), columns=column_values)
    data_test = pd.DataFrame(np.hstack((data_dict["combined_feats_all13"]["X_test"], np.array(y_test).reshape(-1,1), np.array(y_extra_test).reshape(-1,1),test_dummy)),columns=column_values)
    frames = [data_train, data_dev, data_test]
    data_all_features = pd.concat(frames)
    data_all_features.to_csv("{0}_all13features.csv".format(args.name[:-4]), index=False, header=True)
    print("FEATURES SUCCESSFULLY SAVED")
    print("SUCCESS TILL HERE")
    ### DIT MOET OOK VOO RALLE 8 taken aparte naam hebben!!!!!!!!!!!!!!!!!!!!!!!!!!!!!1

    #data_dict = {"mult_negs_feats": {"X_train": get_multiple_negs(X_train), "X_dev": get_multiple_negs(X_dev), "X_test": get_multiple_negs(X_test)}}
    #df = pd.read_csv("all13features.csv", header=0)
    # for item in df.keys():
    # X_train = df[item].iloc[df[set_type] == "train"]
    #data_keys = list(data_dict.keys())

    for feature in data_dict.keys():
        if feature != "mult_negs":
            print(len(X_train), len(y_train))
            X_train = data_dict[feature]["X_train"]
            X_dev = data_dict[feature]["X_dev"]
            X_test = data_dict[feature]["X_test"]
            ngram_ranges = [(1,1),(1,2), (1,3),(1,4),(1,5), (1,6),(1,7),(1,8),(2,3)]
            #ngram_ranges = [(1,6),(1,7),(1,8),(2,3)]
            analyzers = ["word", "char"]
            kernels = ["linear", "rbf"]
            Cs = [1,10,20]
            print("gender: {0}\n".format(gender))
            results = []
            model_file_name = model_name[:-4] + "_" + feature + ".txt" 
            with open(model_file_name, "w", encoding="utf-8") as outfile:
                # for ngram_range in ngram_ranges:
                #     for analyzer in analyzers:
                for kernel in kernels:
                    for C in Cs:
                        #vec = TfidfVectorizer(lowercase=False, ngram_range=ngram_range, analyzer=analyzer) 
                        clf = svm.SVC(kernel=kernel, C=C)
                        #clf = Pipeline( [('vec', vec), ('cls', clf)] )
                        if feature != "combined_feats_all13":
                            X_train = np.array(X_train).reshape(-1, 1)
                            X_dev = np.array(X_dev).reshape(-1,1)
                            X_test = np.array(X_test).reshape(-1,1)
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
                ordered_name = "./ORDERED_{0}_{1}.txt".format(args.name[:-4], feature)
            else:
                if args.medium == "twitter":
                    ordered_name = "./ORDERED_{0}_{1}.txt".format(args.name[:-4], feature)
                else:
                    ordered_name = "./ORDERED_{0}_{1}.txt".format(args.name[:-4], feature)
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
    #print(args.ff)
    if args.countries is not None:
        countries = args.countries.split(",")
    else:
        countries = ["germany", "iran", "italy", "new-delhi", "poland", "portugal", "russia", "spain", "the-netherlands"]

    if args.gender != True:
        twitter_data, full_twitter, medium_data, full_medium = get_nl_data(countries)
    else:
        twitter_data, full_twitter, medium_data, full_medium = get_gender_data(countries)

    ## GRID SEARCH
    if args.cross:
        with open("./logbook.txt", "w", encoding="utf-8") as out:
            out.write("Starting test for cross-genre\n")
            out.write("Gender " + str(args.gender) + "\n")
        #print("Length tuple", str(args.lengths))
        test_svm_params(full_twitter, full_medium, gender=args.gender, downsample=False, within=False, model_name=args.name)
        #get_pos_distro_features(full_twitter, full_medium, within=False)
    else:
        with open("./logbook.txt", "w", encoding="utf-8") as out:
            out.write("Starting test for within-genre\n")
            out.write("Medium" + args.medium + "\n")
            out.write("Gender " + str(args.gender) + "\n")
        #print("Length tuple", str(args.lengths)+"\n")
        print(args.medium+"\n")
        if args.medium == "twitter":
            data = full_twitter
        elif args.medium == "medium":
            data = full_medium
        test_svm_params(data, None, gender=args.gender, downsample=False, within=True, model_name=args.name)
        #get_pos_distro_features(data, None, within=True)

    #cross_val_cross_genre(full_twitter, full_medium, gender=args.gender)


