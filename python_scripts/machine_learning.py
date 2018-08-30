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
from nltk.tokenize import TweetTokenizer
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB
#import matplotlib.pyplot as plt
import nltk

def create_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-l","--length", required=True, type=str, help="Length of each document")
    parser.add_argument("-c","--countries", type=str, help="Which countries (languages) to use")
    parser.add_argument("-p","--path", type=str, help="Where the twitter and medium folders are located")
    parser.add_argument("-cross","--cross", action="store_true", help="Whether it is cross genre or not")
    parser.add_argument("-g","--gender", action="store_true", help="Whether the type is gender or not")
    parser.add_argument("-m","--medium", type=str, help="Which medium if within genre")
    parser.add_argument("-model_name","--model_name", type=str, help="Model name (filename)")
    parser.add_argument("-output_type","--output_type", type=str, help="Output type")
    parser.add_argument("-lengths","--lengths", nargs="+", help="Lengths (tuple)")
    parser.add_argument("-feature_file","--ff", type=str, help="Feature file (pos distro features")
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


def other_ml_algos(X_train, y_train, X_test, y_test, name):
    X_test, y_test, X_dev, y_dev = X_test[:round(len(X_test)*0.5)], y_test[:round(len(X_test)*0.5)], X_test[round(len(X_test)*0.5):], y_test[round(len(X_test)*0.5):]
    vec = TfidfVectorizer()
    classifiers = [MultinomialNB(), DecisionTreeClassifier(), RandomForestClassifier(), MLPClassifier(),KNeighborsClassifier()] 
    with open("results_{0}.txt".format(name), "w", encoding="utf-8") as outfile:
        for classifier in classifiers:
            clf = Pipeline( [('vec', vec), ('cls', classifier)] )
            clf.fit(X_train, y_train)
            y_guess = clf.predict(X_dev)
            accuracy = accuracy_score(y_dev, y_guess)
            f1score = f1_score(y_dev, y_guess, average="macro")
            # print(accuracy, f1score)
            # print(classification_report(y_test, y_guess))
            outfile.write("Results {0}:\n".format(classifier))
            outfile.write("Acc: {0}. F1-score: {1}".format(accuracy, f1score))
            outfile.write("\n\n")


def get_optimal_data_curve(data_large, within=False, gender=False, downsample=False):
    # if cross: data_small and data_large are lists with 2 datasets (Train and test), if within, these are just one dataset

    if within:
        X_train, X_test, y_train, y_test = split_dataset_within(data_large, int(args.length), gender, downsample)
    else:
        X_train, X_test, y_train, y_test = split_dataset_cross(data_large[0], data_large[1], gender, downsample)
        X_train, y_train = shuffle(X_train, y_train, random_state=42)
        X_test, y_test = shuffle(X_test, y_test, random_state=42)

    X_test, y_test, X_dev, y_dev = X_test[:round(len(X_test)*0.5)], y_test[:round(len(X_test)*0.5)], X_test[round(len(X_test)*0.5):], y_test[round(len(X_test)*0.5):]
    initial_length = len(X_train)
    print(len(X_train))
    f1_scores_train = []
    f1_scores_test = []
    percentages = [0.2,0.4,0.6,0.8,1]
    for i in percentages:
        X_train_data, y_train_data, X_dev_data, y_dev_data = X_train[:round(len(X_train)*i)+1], y_train[:round(len(y_train)*i)+1], X_dev[:round(len(X_dev)*i)+1], y_dev[:round(len(y_dev)*i)+1]
        vec = TfidfVectorizer() # add binary=True here
        clf = svm.SVC(kernel="linear")
        clf = Pipeline( [('vec', vec), ('cls', clf)] )
        clf.fit(X_train_data, y_train_data)
        y_guess = clf.predict(X_dev_data)
        y_guess_train = clf.predict(X_train_data)
        accuracy = accuracy_score(y_dev_data, y_guess)
        f1score = f1_score(y_dev_data, y_guess, average="macro")
        f1score_train = f1_score(y_train_data, y_guess_train, average="macro")
        print(accuracy, f1score)
        f1_scores_test.append(f1score)
        f1_scores_train.append(f1score_train)

    plt.plot([item*initial_length for item in percentages], f1_scores_train, "bs", [item*initial_length for item in percentages], f1_scores_test, "r--")
    plt.ylabel("F1-score")
    plt.xlabel("Number of training samples")
    plt.show()

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

def get_lang_specific_features(data_list):
    for item in data:
        sent_length = len(item.split())
        capitalized_words = np.sum([word for word in item if word[0].isupper() and not word[1].isupper()])
        tagged = nltk.pos_tag(item)
        number_of_articles = np.sum([word[0] for word in item if word[1] == "DT"])
        multiple_negatives = True if len([word for word in item if word.lower() == "not" or word.lower() == "n't"]) > 1 else False
        # use spacy for dependency parsing

def get_pos_distribution_features(X_test, y_test, distrofile, model_name="cross_genre", output_type="probs"):
    with open(distrofile, "r", encoding="utf-8") as infile:
        data = infile.readlines()
        X_train = [item.split(",")[0] for item in data]
        y_train = [item.split(",")[1].strip() for item in data]
        #dataset_items = infile_dataset.readlines()

    conversion = {"hi": "new-delhi", "nl": "the-netherlands", "es":"spain", "pt": "portugal", "pl": "poland", "de": "germany", "ru": "russia", "fa":"iran", "it": "italy"}
    y_train = [conversion[item] for item in y_train]
    clf = svm.SVC(kernel="linear", probability=True)
    vect = TfidfVectorizer(ngram_range=(1,3))
    pipeline = Pipeline([("vect", vect), ("clf", clf)])
    pipeline.fit(X_train, y_train)
    ## to do: pickle model
    predictions = pipeline.predict(X_test)
    print("Length", len(X_test))
    if output_type == "probs":
        probabilities = pipeline.predict_proba(X_test)
    else:
        probabilities = pipeline.decision_function(X_test)
    acc = accuracy_score(y_test, predictions)

    print("\n"+str(acc))
    with open(f"{model_name}_{output_type}_distributional_probability_features_pos.csv", "w", encoding="utf-8") as outfile:

        for ix, instance in enumerate(probabilities):
            instance_items = ",".join([str(item) for item in instance])
            #print(instance_items)
            outfile.write(instance_items+","+y_test[ix]+"\n")

def get_pos_distro_features(train, test=None, within=False):
    if within:
        X_train, X_test, y_train, y_test = split_dataset_within(train, int(args.length), False,False)
    else:
        X_train, X_test, y_train, y_test = split_dataset_cross(train, test, False, False)
        X_train, y_train = shuffle(X_train, y_train, random_state=42)
        X_test, y_test = shuffle(X_test, y_test, random_state=42)

    X_train_pos = []
    for item in X_train:
        item_pos = nltk.pos_tag(item.split())
        item_mapped = " ".join([nltk.map_tag("en-ptb", "universal", tag) for word, tag in item_pos])
        X_train_pos.append(item_mapped)

    X_test_pos = []
    for item in X_test:
        item_pos = nltk.pos_tag(item.split())
        item_mapped = " ".join([nltk.map_tag("en-ptb", "universal", tag) for word, tag in item_pos])
        X_test_pos.append(item_mapped)

    X_pos = X_train_pos + X_test_pos
    y_pos = y_train + y_test

    get_pos_distribution_features(X_pos, y_pos, "pos_tag_distributions.txt", model_name=args.model_name, output_type=args.output_type)


def test_svm_params(train, test=None, gender=False, downsample=False, within=False,model_name="", lengths=None):
    # if within:
    #     X_train, X_test, y_train, y_test = split_dataset_within(train, int(args.length), gender, downsample)
    # else:
    #     X_train, X_test, y_train, y_test = split_dataset_cross(train, test, gender, downsample)
    #     X_train, y_train = shuffle(X_train, y_train, random_state=42)
    #     X_test, y_test = shuffle(X_test, y_test, random_state=42)

    # X_test, y_test, X_dev, y_dev = X_test[:round(len(X_test)*0.5)], y_test[:round(len(y_test)*0.5)], X_test[round(len(X_test)*0.5):], y_test[round(len(y_test)*0.5):]
    # keep this order in mind when using the pos-tagging features

    with open(args.feature_file, "r", encoding="utf-8") as infile:
        data = infile.readlines()
        X = [item.split(",")[:-1] for item in data]
        y = [item.split(",")[-1] for item in data]

    X_train = X[:lengths[0]]
    y_train = y[:lengths[0]]
    X_test = X[lengths[0]:lengths[0]+lengths[1]]
    y_test = y[lengths[0]:lengths[0]+lengths[1]]
    X_dev = X[lengths[0]+lengths[1]:]
    y_dev = y[lengths[0]+lengths[1]:]

    assert len(X_train) == lengths[0] and len(X_test) == lengths[1] and len(X_dev) == lengths[2]


    ngram_ranges = [(1,1),(1,2), (1,3),(1,4),(1,5), (1,6),(1,7),(1,8),(2,3)]
    #ngram_ranges = [(1,6),(1,7),(1,8),(2,3)]
    analyzers = ["word", "char"]
    kernels = ["linear"]
    Cs = [1,10,20]
    print("gender: {0}\n".format(gender))
    results = []
    with open(model_name, "w", encoding="utf-8") as outfile, open("./logbook.txt", "a", encoding="utf-8") as outbook:
        for ngram_range in ngram_ranges:
            for analyzer in analyzers:
                for kernel in kernels:
                    for C in Cs:
                        outbook.write("{0} {1} {2} {3}\n".format(ngram_range,  analyzer, kernel, C))
                        vec = TfidfVectorizer(lowercase=False, ngram_range=ngram_range, analyzer=analyzer) 
                        clf = svm.SVC(kernel=kernel, C=C)
                        clf = Pipeline( [('vec', vec), ('cls', clf)] )
                        clf.fit(X_train, y_train)
                        y_guess = clf.predict(X_dev)
                        accuracy = accuracy_score(y_dev, y_guess)
                        f1score = f1_score(y_dev, y_guess, average="macro")
                        outbook.write("{0}\n".format(f1score))
                        outfile.write("SVM with ngram range {0}, analyzer {1}, kernel {2}, C {3}\n".format(ngram_range, analyzer, kernel, C))
                        outfile.write("Accuracy: {0}\n".format(accuracy))
                        outfile.write("(Macro) F1-score: {0}\n".format(f1score))
                        outfile.write(classification_report(y_dev, y_guess))
                        outfile.write("\n\n")
                        results.append(("SVM with ngram range {0}, analyzer {1}, kernel {2}, C {3}\n".format(ngram_range, analyzer, kernel, C), f1score, accuracy))

    if args.cross:
        ordered_name = "./ORDERED_cross_genre_{0}.txt".format(args.output_type)
    else:
        if args.medium == "twitter":
            ordered_name = "./ORDERED_pos_distro_feats_within_genre_twitter_{0}.txt".format(args.output_type)
        else:
            ordered_name = "./ORDERED_pos_distro_feats_within_genre_medium_{0}.txt".format(args.output_type)
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
    if args.countries is not None:
        countries = args.countries.split(",")
    else:
        countries = ["germany", "iran", "italy", "new-delhi", "poland", "portugal", "russia", "spain", "the-netherlands"]




    if args.gender != True:
        twitter_data, full_twitter, medium_data, full_medium = get_nl_data(countries)
    else:
        twitter_data, full_twitter, medium_data, full_medium = get_gender_data(countries)


    print("Stats Twitter")
    for item in full_twitter:
        print(item)
        print("male")
        print(len(full_twitter[item]["male"].split()))
        print("female")
        print(len(full_twitter[item]["female"].split()))


    print("Stats Medium")
    for item in full_medium:
        print(item)
        print("male")
        print(len(full_medium[item]["male"].split()))
        print("female")
        print(len(full_medium[item]["female"].split()))
    

    #print(twitter_data)
    # print("twitter")
    #do_ML_within_genre(full_twitter, int(args.length), True, False, name='twitter')
    # print("medium")
    #do_ML_within_genre(full_medium, int(args.length), True, False, name="medium")
    # print("cross genre (T-M)")
    #do_ML_cross_genre(full_twitter, full_medium, True, False)

    #get_optimal_data_curve([full_twitter, full_medium], within=False, gender=False, downsample=False)

    #X_train, X_dev, y_train, y_dev = split_dataset_cross(full_twitter, full_medium)
    #other_ml_algos(X_train, y_train, X_dev, y_dev, name="cross-genre_nat_lang_no_downsampling")

    
    # get_optimal_data_curve([full_twitter, full_medium], within=False, gender=True, downsample=False)
    # get_optimal_data_curve(full_twitter, within=True, gender=True, downsample=False)
    # get_optimal_data_curve(full_medium, within=True, gender=True, downsample=False)



    ## GRID SEARCH
    # if args.cross:
    #     with open("./logbook.txt", "w", encoding="utf-8") as out:
    #         out.write("Starting test for cross-genre\n")
    #         out.write("Gender " + str(args.gender) + "\n")
    #     print("Length tuple", str(tuple(args.lengths)))
    #     test_svm_params(full_twitter, full_medium, gender=args.gender, downsample=False, within=False, name=args.model_name, lengths=tuple(args.lengths))
    #     #get_pos_distro_features(full_twitter, full_medium, within=False)
    # else:
    #     with open("./logbook.txt", "w", encoding="utf-8") as out:
    #         out.write("Starting test for within-genre\n")
    #         out.write("Medium" + args.medium + "\n")
    #         out.write("Gender " + str(args.gender) + "\n")
    #     print("Length tuple", str(tuple(args.lengths))+"\n")
    #     print(args.medium+"\n")
    #     if args.medium == "twitter":
    #         data = full_twitter
    #     elif args.medium == "medium":
    #         data = full_medium
    #     test_svm_params(data, None, gender=args.gender, downsample=False, within=True, model_name=args.model_name, lengths=tuple(args.lengths))
    #     #get_pos_distro_features(data, None, within=True)

    #cross_val_cross_genre(full_twitter, full_medium, gender=args.gender)


    ## run on server (maybe at label in txt file?)
    ## run within-genre on server
    ## use as features on server

    # lengths cross-genre
    # Train 1775
    # Test 957
    # Dev 957

    # lengths twitter within-genre
    # Train 1189
    # Test 293
    # Dev 293

    # lengths medium within genre
    # Train 1282
    # Test 316
    # Dev 316

