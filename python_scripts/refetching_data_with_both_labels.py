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
    parser.add_argument("-m","--medium", required=True, type=str, help="Which medium if within genre")
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
    

def get_data(path, countries, affix, excel_file):
    df = pd.read_excel(excel_file)
    folders = [name for name in os.listdir(path) if os.path.isdir(os.path.join(path, name)) and name in countries]
    dataset = defaultdict(dict)
    for folder in folders:
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
                    with open(os.path.join(path+folder+"/",file), "r", encoding="utf-8") as infile:
                        data = infile.read().split("\n\n")
                        for line in data:
                            if gender not in dataset[folder]:
                                    dataset[folder][gender] = preprocess_text(line.strip(), affix, folder, pos_tagging=False)
                            else:
                                dataset[folder][gender] += " " + preprocess_text(line.strip(), affix, folder, pos_tagging=False)
                except ValueError:
                    pass

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
        for country, v in data.items():
            for gndr,val in v.items():
                portions = split_data(val, length)
                for portion in portions:
                    X.append(portion)
                    y.append(country+"_"+gndr)
    else:
        for country,v in data.items():
            for gndr,val in v.items():
                portions = split_data(val, length)
                for portion in portions:
                    X.append(portion)
                    y.append(gndr+"_"+country)

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
        for country,v in train.items():
            for gndr, val in v.items():
                portions = split_data(val, int(args.length))
                for portion in portions:
                    X_train.append(portion)
                    y_train.append(country+"_"+gndr)

        for country,v in test.items():
            for gndr, val in v.items():
                portions = split_data(val, int(args.length))
                for portion in portions:
                    X_test.append(portion)
                    y_test.append(country+"_"+gndr)

    else:
        for country,v in train.items():
            for gndr,val in v.items():
                portions = split_data(val, int(args.length))
                for portion in portions:
                    X_train.append(portion)
                    y_train.append(gndr+"_"+country)
        for country,v in test.items():
            for gndr,val in v.items():
                portions = split_data(val, int(args.length))
                for portion in portions:
                    X_test.append(portion)
                    y_test.append(gndr+"_"+country)

    if downsample:
        X_train, y_train = downsample_data(X_train,y_train)
        X_test, y_test = downsample_data(X_test,y_test)

    return X_train, X_test, y_train, y_test

def merge_datadicts_nl(dict1, dict2):
    ## dict2 is smallest
    dict3 = defaultdict(dict)

    for country,v in dict1.items():
        for gender,val in v.items():
            dict3[country][gender] = val
            if gender in dict2[country]:
                dict3[country][gender] += dict2[country][gender]

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
    twitter_data = get_data(args.path+"/twitter/", countries, "_messages.txt", "./results/META_DATA_CORPUS/data_twitter_excel_filtered.xlsx")
    # print("Twitter")
    # do_ML_within_genre(twitter_data, int(args.length))
    more_twitter_data = get_data(args.path+"/twitter_additional_data/", countries, "_messages.txt", "./results/META_DATA_CORPUS/data_twitter_excel_additional_data.xlsx")
    full_twitter = merge_datadicts_nl(twitter_data, more_twitter_data)
    medium_data = get_data(args.path+"/medium/", countries, "messages_@", "./results/META_DATA_CORPUS/data_medium_excel_filtered.xlsx")
    # print("Medium")
    # do_ML_within_genre(medium_data, int(args.length))
    more_medium_data = get_data(args.path+"/medium_additional_data/", countries, "messages_@", "./results/META_DATA_CORPUS/data_medium_excel_additional_data.xlsx")
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

def stacking(train, test, within=False, gender=False):
    ### cross-val for training features > probabilities
    ### predictions based on training data for dev > probabilities
    ### maybe already get test predictions > probabilities too?
    ### do this for both ngrams + linguistic and then add together 9 + 9
    pass


def load_data(data_file):
    datafile = pd.read_csv(data_file, header=None)
    train = datafile.loc[datafile.iloc[:,-1] == "train"]
    test = datafile.loc[datafile.iloc[:,-1] == "test"]
    dev = datafile.loc[datafile.iloc[:,-1] == "dev"]
    X_train = [item[0] for item in train.iloc[:,:-2].values]
    y_train = train.iloc[:,-2]
    X_dev = [item[0] for item in dev.iloc[:,:-2].values]
    y_dev = dev.iloc[:,-2]
    X_test = [item[0] for item in test.iloc[:,:-2].values]
    y_test = test.iloc[:,-2]
    return X_train, y_train, X_dev, y_dev, X_test, y_test

def test_svm_params(train, test=None, gender=False, downsample=False, within=False, model_name=""):
    if within:
        X_train, X_test, y_train, y_test = split_dataset_within(train, int(args.length), gender, downsample)
    else:
        X_train, X_test, y_train, y_test = split_dataset_cross(train, test, gender, downsample)
        X_train, y_train = shuffle(X_train, y_train, random_state=42)
        X_test, y_test = shuffle(X_test, y_test, random_state=42)

    real_y_train = [item.split("_")[0] for item in y_train]
    other_y_train_label = [item.split("_")[1] for item in y_train]
    real_y_test = [item.split("_")[0] for item in y_test]
    other_y_test_label = [item.split("_")[1] for item in y_test]
    y_train = real_y_train
    y_test = real_y_test



    X_test, y_test, X_dev, y_dev = X_test[:round(len(X_test)*0.5)], y_test[:round(len(y_test)*0.5)], X_test[round(len(X_test)*0.5):], y_test[round(len(y_test)*0.5):]
    # #keep this order in mind when using the pos-tagging features

    other_y_test_label, other_y_dev_label = other_y_test_label[:round(len(other_y_test_label)*0.5)], other_y_test_label[round(len(other_y_test_label)*0.5):]

    train_dummy = np.array(["train" for item in y_train]).reshape(-1,1)
    dev_dummy = np.array(["dev" for item in y_dev]).reshape(-1,1)
    test_dummy = np.array(["test" for item in y_test]).reshape(-1,1)

    print(X_train[0])
    print(np.array(y_train).reshape(-1,1)[0])
    print(np.array(other_y_train_label).reshape(-1,1)[0])
    print(train_dummy[0])

    ## we need
    ## X train, X test, X dev in one coolumn, y_label in the next,  second label in the third, dummies in the fourth
    data_train = pd.DataFrame(np.hstack((np.array(X_train).reshape(-1,1), np.array(y_train).reshape(-1,1), np.array(other_y_train_label).reshape(-1,1),train_dummy)), columns=["text", "label", "second_label", "dataset_type"])
    data_dev = pd.DataFrame(np.hstack((np.array(X_dev).reshape(-1,1), np.array(y_dev).reshape(-1,1), np.array(other_y_dev_label).reshape(-1,1),dev_dummy)), columns=["text", "label", "second_label", "dataset_type"])
    data_test = pd.DataFrame(np.hstack((np.array(X_test).reshape(-1,1), np.array(y_test).reshape(-1,1), np.array(other_y_test_label).reshape(-1,1),test_dummy)), columns=["text", "label", "second_label", "dataset_type"])
    frames = [data_train, data_dev, data_test]
    data = pd.concat(frames)
    data.to_csv(args.name, index=False, header=True)

    #X_train, y_train, X_dev, y_dev, X_test, y_test = load_data(args.data_file)
    ## todo kijken wat volgorde is van nieuwe feature bestanden, nog steeds anders? dan opnieuw features extracten voor word order maar dit keer door dit bestand te laden

    
    # data_train = pd.DataFrame(np.hstack((data_dict["combined_feats_all13"]["X_train"], np.array(y_train).reshape(-1,1), train_dummy)), columns=["sent_length", "capitalization", "articles", "mult_negs", "word_order", "color", "filler", "hedges", "swearwords", "sentiment", "min_response", "adjectives", "punctuation", "set_type", "y_label"])
    # data_dev = pd.DataFrame(np.hstack((data_dict["combined_feats_all13"]["X_dev"], np.array(y_dev).reshape(-1,1), dev_dummy)), columns=["sent_length", "capitalization", "articles", "mult_negs", "word_order", "color", "filler", "hedges", "swearwords", "sentiment", "min_response", "adjectives", "punctuation", "set_type", "y_label"])
    # data_test = pd.DataFrame(np.hstack((data_dict["combined_feats_all13"]["X_test"], np.array(y_test).reshape(-1,1), test_dummy)),columns=["sent_length", "capitalization", "articles", "mult_negs", "word_order", "color", "filler", "hedges", "swearwords", "sentiment", "min_response", "adjectives", "punctuation", "set_type", "y_label"])
    # frames = [data_train, data_dev, data_test]
    # data_all_features = pd.concat(frames)
    # data_all_features.to_csv("all13features.csv", index=False, header=0)

    ## word order == 6 "has_svo", "has_sov", "has_vos", "has_vso", "has_ovs", "has_osv", feats, colors == 2 "special_color", "regular_color",, punct == 2 "question", "exclamation"
    ## nargs

    # nlp = spacy.load('en_core_web_lg')
    # X_train_features = np.array(get_word_order(X_train, nlp))
    # X_dev_features = np.array(get_word_order(X_dev, nlp))
    # X_test_features = np.array(get_word_order(X_test, nlp))
    # train_dummy = np.array(["train" for item in y_train]).reshape(-1,1)
    # dev_dummy = np.array(["dev" for item in y_dev]).reshape(-1,1)
    # test_dummy = np.array(["test" for item in y_test]).reshape(-1,1)
    # data_train = pd.DataFrame(np.hstack((X_train_features, np.array(y_train).reshape(-1,1), train_dummy)))
    # data_dev = pd.DataFrame(np.hstack((X_dev_features, np.array(y_dev).reshape(-1,1), dev_dummy)))
    # data_test = pd.DataFrame(np.hstack((X_test_features, np.array(y_test).reshape(-1,1), test_dummy)))
    # frames = [data_train, data_test, data_dev]
    # data = pd.concat(frames)
    # data.to_csv(args.name, index=False, header=False)

    # # X_train = get_lang_specific_features(X_train)
    # # X_dev = get_lang_specific_features(X_dev)
    # # X_test = get_lang_specific_features(X_test)


    



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
        if args.medium == "twitter":
            test_svm_params(full_twitter, full_medium, gender=args.gender, downsample=False, within=False, model_name=args.name)
        elif args.medium == "medium":
            test_svm_params(full_medium, full_twitter, gender=args.gender, downsample=False, within=False, model_name=args.name)
        #get_pos_distro_features(full_twitter, full_medium, within=False)
    else:
       
        if args.medium == "twitter":
            data = full_twitter
        elif args.medium == "medium":
            data = full_medium
        test_svm_params(data, None, gender=args.gender, downsample=False, within=True, model_name=args.name)
        #get_pos_distro_features(data, None, within=True)

    #cross_val_cross_genre(full_twitter, full_medium, gender=args.gender)



    ## NL aanpasssen zodat gender bekend is 
    ## opnieuw data files maken met alle data waarin volgorde de nieuwe instellingen meenemen + nl en gender staan er gewoon in
    ## eerst word order feature extraction opnieuw runnen met nieuwe data
    ## met die nieuwe data files de lang specific ml doen, voor alles maar ff opnieuw


