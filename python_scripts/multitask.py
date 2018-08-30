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
from keras.layers import Embedding, Dropout, LSTM, Bidirectional, Input, BatchNormalization, merge, concatenate
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder
np.random.seed(42)

def create_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-name","--name", type=str, help="Model name (filename)")
    parser.add_argument("-data_file","--data_file", type=str, help="Data file")
    parser.add_argument("-cross", "--cross", action="store_true", help="Use if cross-genre is true")
    parser.add_argument("-gender", "--gender", action="store_true", help="Use if gender is true")
    parser.add_argument("-json_emb", "--json_emb", type=str, help="JSON embeddings")
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
    batch_size = 250
    ep = 1
    batch_sizes = [50, 150, 250]
    epochs = [1,3,6,10]
    epochs_two = [15, 20, 25, 30]
    epochs_three = [40, 50]

    if args.cross and "native_lang_tm" in args.name:
        batch_size = 150
        epochs = 30
    elif args.cross and "native_lang_mt" in args.name:
        batch_size = 150
        epochs = 3
    elif "native_lang_twitter" in args.name:
        batch_size = 50
        epochs = 30
    elif "native_lang_medium" in args.name:
        batch_size = 50
        epochs = 50
    elif args.cross and "gender_tm" in args.name:
        batch_size = 50
        epochs = 3
    elif args.cross and "gender_mt" in args.name:
        batch_size = 50
        epochs = 15
    elif not args.cross and "gender_twitter" in args.name:
        batch_size = 50
        epochs = 40
    elif not args.cross and "gender_medium" in args.name:
        batch_size = 50
        epochs = 40

    

    with open("test_results_"+ args.name + ".csv", "w", encoding="utf-8") as outfile:
        print(args.data_file)
        outfile.write(args.data_file+"\n")
        print('Build models...')

        main_input = Input(shape=(input_length,), dtype='float32', name='main_input')
        x = Embedding(input_dim = vocab_size, output_dim = dim, weights=[embedding_matrix], input_length=input_length, trainable=True)(main_input)
        lstm = Bidirectional(LSTM(output_dim = 50, input_dim = dim, dropout=0.3, recurrent_dropout=0.3) )(x)

        aux_input = Input(shape=(input_length,), dtype="float32", name="aux_input")
        auxiliary_input = Embedding(input_dim = vocab_size, output_dim = dim, weights=[embedding_matrix], input_length=input_length, trainable=True)(aux_input)
        t_auxiliary_input = Bidirectional(LSTM(output_dim=50, input_dim=dim, dropout=0.3, recurrent_dropout=0.3))(auxiliary_input)


        x = concatenate([lstm, t_auxiliary_input])

        x = Dense(60, activation='tanh')(x)
        ## add lstm layer here? test 
        x = Dropout(0.5)(x)


        if not args.gender:
            task1_output = Dense(9, activation='softmax', name='main_output')(x)
            task2_output = Dense(2, activation='softmax', name='aux_output')(x)
        else:
            task1_output = Dense(2, activation='softmax', name='main_output')(x)
            task2_output = Dense(9, activation='softmax', name='aux_output')(x)

        model_task1 = Model(input=[main_input, aux_input], output=[task1_output, task2_output])
        #model_task2 = Model(input=[main_input, aux_input], output=[task2_output])

        model_task1.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'], loss_weights=[1., 0.4])
        #model_task2.compile(optimizer='RMSprop', loss='categorical_crossentropy', metrics=['accuracy'])
        print(model_task1.summary())
        #print(model_task2.summary())
        h = model_task1.fit([padded_docs, padded_docs], [labels, y_extra_train], batch_size=batch_size, epochs=epochs, verbose=1)
        #h2 = model_task2.fit([padded_docs, padded_docs], y_extra_train, batch_size=batch_size, epochs=ep, verbose=0)

        scores = model_task1.evaluate([X_test, X_test], [y_test, y_extra_test], verbose=1)
        main = scores[-2]
        aux = scores[-1]
        print("Test Scores: {0},{1}, with batch size {2} and epochs {3}\n".format(str(main), str(aux), batch_size, epochs))
        outfile.write("Test Scores: {0}, {1}, with batch size {2} and epochs {3}\n".format(str(main), str(aux), batch_size, epochs))

            #print(accuracy, accuracy2)
            # loss2, accuracy2 = model_task2.evaluate([X_test, X_test], y_extra_test, verbose=0)
            # print(accuracy2)



if __name__ == "__main__":
    args = create_arg_parser()

    X_train, y_train, X_dev, y_dev, X_test, y_test, y_extra_train, y_extra_dev, y_extra_test = load_data(args.data_file, header_present=[0])
    t, vocab_size, padded_docs, encoded_docs, max_length, X_test = process_data(X_train, X_test)
    dim = int(args.dim)
    with open(args.json_emb, "r", encoding="utf-8") as json_file:
        embeddings_index = json.load(json_file)
        print(len(embeddings_index))

    run_multitask_model(vocab_size, t, embeddings_index, max_length, dim, padded_docs, y_train, X_test, y_test, y_extra_train, y_extra_test)


    #python python_scripts/multitask.py -cross -data_file my_data/data_final_cross_genre_native_lang_tm.csv -json_emb regular_pretrained_embeddings_control_25d.json -dim 25  