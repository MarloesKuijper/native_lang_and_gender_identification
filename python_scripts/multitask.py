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
from keras.models import Sequential, Model
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Embedding, Dropout, LSTM, Bidirectional, Input, BatchNormalization, merge
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder
np.random.seed(42)

def create_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-name","--name", type=str, help="Model name (filename)")
    parser.add_argument("-data_file","--data_file", type=str, help="Data file")
    parser.add_argument("-cross", "--cross", action="store_true", help="Use if cross-genre is true")
    parser.add_argument("-gender", "--gender", action="store_true", help="Use if gender is true")
    parser.add_argument("-json_emb", "--json_emb", type=str, help="JSON emebddings")
    parser.add_argument("-dim", "--dim", required=True, type=str, help="Dimensions")
    # parser.add_argument("-type", "--type", required=True, type=str, help="Type: lstm or bilstm")
    # parser.add_argument("-batch_size", "--batch_size", required=True, type=str, help="Dimensions")
    # parser.add_argument("-epochs", "--epochs", required=True, type=str, help="epochs")
    # parser.add_argument("-opt", "--opt", required=True, type=str, help="optimizer")
    # parser.add_argument("-neurons", "--neurons", required=True, type=str, help="neurons")
    # parser.add_argument("-dropout", "--dropout", required=True, type=str, help="dropout")
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
    y_extra_train = train.iloc[:,-2]
    y_extra_dev = dev.iloc[:,-2]
    y_extra_test = test.iloc[:,-2]
    
    return X_train, y_train, X_dev, y_dev, X_test, y_test, y_extra_train, y_extra_dev, y_extra_test


def process_data(X_train, X_test):
    X_train = [" ".join(word_tokenize(item)) for item in X_train]
    X_test = [" ".join(word_tokenize(item)) for item in X_test]
    t = Tokenizer()
    t.fit_on_texts(X_train)
    vocab_size = len(t.word_index) + 1
    encoded_docs = t.texts_to_sequences(X_train)
    max_length = 300
    padded_docs = pad_sequences(encoded_docs, maxlen=max_length, padding='post')

    X_test_encoded = t.texts_to_sequences(X_test)
    X_test_padded = pad_sequences(X_test_encoded, maxlen=max_length, padding="post")
    return t, vocab_size, padded_docs, encoded_docs, max_length, X_test_padded


def run_multitask_model(vocab_size, t, embeddings_index, input_length, dim, padded_docs, labels, X_test, y_test, y_extra_train, y_extra_test):
    le = LabelEncoder()
    le.fit(labels)
    labels = le.transform(labels)
    print(labels[0])
    labels = np_utils.to_categorical(labels)
    print(labels[0])
    y_test = le.transform(y_test)
    y_test = np_utils.to_categorical(y_test)

    le2 = LabelEncoder()
    le2.fit(y_extra_train)
    y_extra_train = le2.transform(y_extra_train)
    y_extra_train = np_utils.to_categorical(y_extra_train)
    y_extra_test = le2.transform(y_extra_test)
    y_extra_test = np_utils.to_categorical(y_extra_test)

    embedding_matrix = zeros((vocab_size, dim))
    for word, i in t.word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = np.array(embedding_vector, dtype="float32")
    # define model
    batch_size = 150
    ep = 10
    print('Build models...')

    main_input = Input(shape=(input_length,), dtype='float32', name='main_input')

    x = Embedding(input_dim = vocab_size, output_dim = dim, weights=[embedding_matrix], input_length=input_length, trainable=True, dropout=0.3)(main_input)
    lstm = Bidirectional(LSTM(output_dim = 50, input_dim = dim, dropout_W=0.3, dropout_U=0.3) )(x)
    lstm_out = Dropout(0.1)(lstm)

    aux_input = Input(shape=(input_length,), dtype="float32", name="aux_input")
    auxiliary_input = Embedding(input_dim = vocab_size, output_dim = dim, weights=[embedding_matrix], input_length=input_length, trainable=True, dropout=0.3)(aux_input)
    t_auxiliary_input = LSTM(output_dim=50, input_dim=dim, dropout_W=0.3, dropout_U=0.3)(auxiliary_input)

    x = merge([lstm_out, t_auxiliary_input], mode='concat')

    x = Dense(30, activation='tanh', )(x)
    x = Dropout(0.5)(x)

    task1_output = Dense(9, activation='softmax', name='main_output')(x)
    task2_output = Dense(2, activation='softmax', name='aux_output')(x)


    model_task1 = Model(input=[main_input, aux_input], output=[task1_output])
    model_task2 = Model(input=[main_input, aux_input], output=[task2_output])

    model_task1.compile(optimizer='RMSprop', loss='categorical_crossentropy', metrics=['accuracy'])
    model_task2.compile(optimizer='RMSprop', loss='categorical_crossentropy', metrics=['accuracy'])
    print(model_task1.summary())
    print(model_task2.summary())
    h = model_task1.fit([padded_docs, padded_docs], labels, batch_size=batch_size, epochs=ep, verbose=0)
    h2 = model_task2.fit([padded_docs, padded_docs], y_extra_train, batch_size=batch_size, epochs=ep, verbose=0)

    loss, accuracy = model_task1.evaluate([X_test, X_test], y_test, verbose=0)
    print(accuracy)
    loss2, accuracy2 = model_task2.evaluate([X_test, X_test], y_extra_test, verbose=0)
    print(accuracy2)



if __name__ == "__main__":
    args = create_arg_parser()

    X_train, y_train, X_dev, y_dev, X_test, y_test, y_extra_train, y_extra_dev, y_extra_test = load_data(args.data_file, header_present=[0])
    t, vocab_size, padded_docs, encoded_docs, max_length, X_dev = process_data(X_train, X_dev)
    dim = int(args.dim)
    with open(args.json_emb, "r", encoding="utf-8") as json_file:
        embeddings_index = json.load(json_file)
        print(len(embeddings_index))

    run_multitask_model(vocab_size, t, embeddings_index, max_length, dim, padded_docs, y_train, X_dev, y_dev, y_extra_train, y_extra_dev)


    #python python_scripts/multitask.py -cross -data_file my_data/data_final_cross_genre_native_lang_tm.csv -json_emb concatenated_embeddings_25d_pca.json -dim 25  