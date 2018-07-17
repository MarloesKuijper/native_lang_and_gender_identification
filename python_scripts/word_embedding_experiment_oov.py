## regular english: Twitter (2B tweets, 27B tokens, 1.2M vocab, uncased, 25d, 50d, 100d, & 200d vectors, 1.42 GB download): glove.twitter.27B.zip
## learner english: 5000+ sentences from Cambridge Learner Corpus (cite) + my own dataset

from nltk.tokenize import TweetTokenizer, word_tokenize, sent_tokenize
import argparse, os
from gensim.models import Word2Vec
from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.models import KeyedVectors
from collections import defaultdict
import pandas as pd
import numpy as np
import json, re
from numpy import array
from numpy import asarray
from numpy import zeros
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Embedding, Dropout, LSTM
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder
np.random.seed(42)

def create_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-data_folder","--data_folder", type=str, help="Folder with my own data")
    parser.add_argument("-name","--name", type=str, help="Model name (filename)")
    parser.add_argument("-data_file","--data_file", type=str, help="Data file")
    parser.add_argument("-cross", "--cross", action="store_true", help="Use if cross-genre is true")
    parser.add_argument("-gender", "--gender", action="store_true", help="Use if gender is true")
    parser.add_argument("-cambridge", "--cambridge", type=str, help="Cambridge sentences")
    parser.add_argument("-wer_txt", "--wer_txt", type=str, help="Pretrained regular embeddings (txt)")
    parser.add_argument("-wer_json", "--wer_json", type=str, help="Pretrained regular embeddings as dict (json)")
    parser.add_argument("-wel_txt", "--wel_txt", type=str, help="Learner embeddings (txt)")
    parser.add_argument("-wel_json", "--wel_json", type=str, help="Learner embeddings as dict (json)")
    parser.add_argument("-type", "--type", required=True, type=str, help="Pick the type: regular, learner, retro")
    parser.add_argument("-dim", "--dim", required=True, type=str, help="Dimensions")
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

def preprocess_sent(sentence):
    #tokenizer = TweetTokenizer(reduce_len=True, preserve_case=False)
    cleanr = re.compile('<.*?>')
    remove_markup = re.sub(cleanr, '', sentence)
    replace_urls = re.sub(r"http\S+", "URL", remove_markup)
    replace_digits = re.sub(r'\d+', "NUM", replace_urls)
    # chose to lowercase here because of sparsity and small embedding set
    return replace_digits.lower()

def get_data(path, countries, affix):
    folders = [name for name in os.listdir(path) if os.path.isdir(os.path.join(path, name)) and name in countries]
    dataset = defaultdict(str)
    full_set_of_sentences = []
    for folder in folders:
        contents = os.listdir(os.path.join(path,folder))
        for file in contents:
            if file.endswith(affix) or file.startswith(affix):
                with open(os.path.join(path+folder+"/",file), "r", encoding="utf-8") as infile:
                    data = infile.read().split("\n\n")
                    for text in data:
                        sentences = sent_tokenize(text)
                        tokenized_sents = [word_tokenize(preprocess_sent(sentence)) for sentence in sentences]
                        for item in tokenized_sents:
                            if item:
                                full_set_of_sentences.append(item)


    #print(dataset["the-netherlands"])
    return full_set_of_sentences


def create_learner_embeddings(sentence_path, corpus_data, dim, pretrained=""):
    with open(sentence_path, "r", encoding="utf-8") as infile:
        data = [word_tokenize(sentence.strip().lower()) for sentence in infile.readlines()]
    data_complete = data + corpus_data
    model = Word2Vec(data_complete, min_count=1, size=dim)
    total_examples = model.corpus_count
    print(total_examples)
    if pretrained:
        glove2word2vec(pretrained, "pretrained_glove25d.txt.word2vec")
        model_pretrained = KeyedVectors.load_word2vec_format("pretrained_glove25d.txt.word2vec", binary=False)
        model.build_vocab([list(model_pretrained.vocab.keys())], update=True)
        model.intersect_word2vec_format("pretrained_glove25d.txt.word2vec", binary=False, lockf=1.0)
        model.train(data_complete, total_examples=total_examples, epochs=model.iter)

    print("new count", model.corpus_count)
    words = list(model.wv.vocab)
    print(len(words))
    # model.save("model.bin")
    # model = KeyedVectors.load_word2vec_format('model.bin', binary=True, unicode_errors="ignore")
    model.wv.save_word2vec_format(args.wel_txt, binary=False)
    # model = KeyedVectors.load_word2vec_format("model_learner_english.txt", binary=False)
    # print(model)
    ##load model
    # loaded_model = Word2Vec.load("model.bin")
    return model

def load_regular_pretrained_glove_embeddings(pretrained_embeddings):
    word2vec_output_file = 'output.txt.word2vec'
    glove2word2vec(pretrained_embeddings, word2vec_output_file)
    # load the Stanford GloVe model
    filename = 'output.txt.word2vec'
    model = KeyedVectors.load_word2vec_format(filename, binary=False, limit=89503)
    # calculate: (king - man) + woman = ?
    result = model.most_similar(positive=['woman', 'king'], negative=['man'], topn=1)
    print(result)
    model.wv.save_word2vec_format("regular_pretrained_embeddings_controlled_for_size_100d.txt", binary=False)

def run_model(vocab_size, t, embeddings_index, input_length, dim, padded_docs, labels, X_test, y_test):
    # create a weight matrix for words in training docs
    le = LabelEncoder()
    le.fit(labels)
    labels = le.transform(labels)
    print(labels[0])
    labels = np_utils.to_categorical(labels)
    print(labels[0])
    y_test = le.transform(y_test)
    y_test = np_utils.to_categorical(y_test)

    embedding_matrix = zeros((vocab_size, dim))
    for word, i in t.word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = np.array(embedding_vector, dtype="float32")
        else:
            pass
            #print(word)
    # define model
    model = Sequential()
    e = Embedding(vocab_size, dim, weights=[embedding_matrix], input_length=input_length, trainable=False)
    model.add(e)
    model.add(Flatten())
    if args.gender:
        model.add(Dense(2, activation='softmax'))
    else:
        model.add(Dense(9, activation='softmax'))
    # compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
    # summarize the model
    print(model.summary())
    # fit the model
    model.fit(padded_docs, labels, batch_size=16, epochs=50, verbose=0)
    #weights = model.layers[0].get_weights()[0]

    # # evaluate the model
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print("Accuracy: {0}".format(accuracy))
    with open("results_word_embedding_experiment_{0}d_ep50_bs16_adam_mlp_retro_oov.txt".format(dim), "a", encoding="utf-8") as outfile:
        outfile.write("Filename: {0}".format(args.name[:-4]+"\n"))
        outfile.write("Accuracy: {0}, Loss: {1}\n".format(accuracy, loss))

def run_lstm_model(vocab_size, t, embeddings_index, input_length, dim, padded_docs, labels, X_test, y_test):
    # create a weight matrix for words in training docs
    le = LabelEncoder()
    le.fit(labels)
    labels = le.transform(labels)
    print(labels[0])
    labels = np_utils.to_categorical(labels)
    print(labels[0])
    y_test = le.transform(y_test)
    y_test = np_utils.to_categorical(y_test)

    embedding_matrix = zeros((vocab_size, dim))
    for word, i in t.word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = np.array(embedding_vector, dtype="float32")
    # define model
    model = Sequential()
    e = Embedding(vocab_size, dim, weights=[embedding_matrix], input_length=input_length, trainable=False)
    model.add(e)
    model.add(LSTM(128))
    model.add(Dropout(0.5))
    if args.gender:
        model.add(Dense(2, activation='softmax'))
    else:
        model.add(Dense(9, activation='softmax'))
    # compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
    # summarize the model
    print(model.summary())
    # fit the model
    model.fit(padded_docs, labels, batch_size=16, epochs=50, verbose=0)
    # evaluate the model
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print("Accuracy: {0}".format(accuracy))
    with open("results_word_embedding_experiment_{0}d_ep50_bs16_adam_lstm_retro_oov.txt".format(dim), "a", encoding="utf-8") as outfile:
        outfile.write("Filename: {0}".format(args.name[:-4]+"\n"))
        outfile.write("Accuracy: {0}, Loss: {1}\n".format(accuracy, loss))

def process_data(X_train, X_test):
    X_train = [" ".join(word_tokenize(item)) for item in X_train]
    X_test = [" ".join(word_tokenize(item)) for item in X_test]
    # prepare tokenizer
    t = Tokenizer()
    t.fit_on_texts(X_train)
    vocab_size = len(t.word_index) + 1
    # integer encode the documents
    encoded_docs = t.texts_to_sequences(X_train)
    #print(encoded_docs)
    # pad documents to a max length of 4 words
    max_length = 300
    padded_docs = pad_sequences(encoded_docs, maxlen=max_length, padding='post')
    #print(padded_docs)

    X_test_encoded = t.texts_to_sequences(X_test)
    X_test_padded = pad_sequences(X_test_encoded, maxlen=max_length, padding="post")
    return t, vocab_size, padded_docs, encoded_docs, max_length, X_test_padded

def create_embeddings_dict(embeddings_path, embeddings_name):
    # load the whole embedding into memory
    embeddings_index = dict()
    with open(embeddings_path, "r", encoding='utf-8') as file:
        for line in file.readlines():
            values = line.strip().split()
            word = values[0]
            coefs = values[1:]
            if len(coefs) < 2:
                print(line)
                pass
            else:
                embeddings_index[word] = coefs
    print('Loaded %s word vectors.' % len(embeddings_index))
    with open(embeddings_name, "w", encoding="utf-8") as outfile:
        json.dump(embeddings_index, outfile)

    return embeddings_index

if __name__ == "__main__":
    args = create_arg_parser()
    ## 0. Process data
    X_train, y_train, X_dev, y_dev, X_test, y_test = load_data(args.data_file, header_present=[0])
    t, vocab_size, padded_docs, encoded_docs, max_length, X_dev = process_data(X_train, X_dev)
    dim = int(args.dim)
    # 1. Run regular embeddings (tweak the number of dimensions, currently 25D)
    if args.type == "regular":
        #load_regular_pretrained_glove_embeddings(args.wer_txt)
        #embeddings_index = create_embeddings_dict(args.wer_txt, args.wer_json)
        with open(args.wer_json, "r", encoding="utf-8") as json_file:
            embeddings_index = json.load(json_file)
            print(len(embeddings_index))
        run_model(vocab_size, t, embeddings_index, max_length, dim, padded_docs, y_train, X_dev, y_dev)
        run_lstm_model(vocab_size, t, embeddings_index, max_length, dim, padded_docs, y_train, X_dev, y_dev)
     
    elif args.type == "learner":
        ## 2. Run learner embeddings, use same Dimensions and same vocab size if possible (or reduce the vocab size of previous)
        # countries = ["germany", "iran", "italy", "new-delhi", "poland", "portugal", "russia", "spain", "the-netherlands"]
        # twitter = get_data(args.data_folder+"/twitter/", countries, "_messages.txt")
        # twitter_2 = get_data(args.data_folder+"/twitter_additional_data/", countries, "_messages.txt")
        # medium = get_data(args.data_folder+"/medium/", countries, "messages_@")
        # medium_2 = get_data(args.data_folder+"/medium_additional_data/", countries, "messages_@")
        # corpus_data = twitter + twitter_2 + medium + medium_2
        # model = create_learner_embeddings(args.cambridge, corpus_data, dim, args.wer_txt)
        # # model = KeyedVectors.load_word2vec_format(args.wel_txt, binary=False)
        # # print(model.wv.vocab)
        #embeddings_index = create_embeddings_dict(args.wel_txt, args.wel_json)
        with open(args.wel_json, "r", encoding="utf-8") as json_file:
            embeddings_index = json.load(json_file)
            print(len(embeddings_index))
        run_model(vocab_size, t, embeddings_index, max_length, dim, padded_docs, y_train, X_dev, y_dev)
        run_lstm_model(vocab_size, t, embeddings_index, max_length, dim, padded_docs, y_train, X_dev, y_dev)

    elif args.type == "concat":
        print("running concat")
        with open(args.wer_json, "r", encoding="utf-8") as json_file:
            embeddings_index = json.load(json_file)
            print(len(embeddings_index))
        run_model(vocab_size, t, embeddings_index, max_length, dim+dim, padded_docs, y_train, X_dev, y_dev)
        ## next time, check out error with h5py?
    # 
    ## word count 89503