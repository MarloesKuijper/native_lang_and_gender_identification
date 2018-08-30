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
from keras.layers import Embedding, Dropout, LSTM, Bidirectional
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
np.random.seed(42)

def create_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-name","--name", type=str, help="Model name (filename)")
    parser.add_argument("-data_file","--data_file", type=str, help="Data file")
    parser.add_argument("-cross", "--cross", action="store_true", help="Use if cross-genre is true")
    parser.add_argument("-gender", "--gender", action="store_true", help="Use if gender is true")
    parser.add_argument("-json_emb", "--json_emb", type=str, help="JSON emebddings")
    parser.add_argument("-dim", "--dim", required=True, type=str, help="Dimensions")
    parser.add_argument("-type", "--type", required=True, type=str, help="Type: lstm or bilstm")
    parser.add_argument("-batch_size", "--batch_size",  type=int, help="Dimensions")
    parser.add_argument("-epochs", "--epochs", type=int, help="epochs")
    parser.add_argument("-opt", "--opt", type=str, help="optimizer")
    parser.add_argument("-neurons", "--neurons", type=int, help="neurons")
    parser.add_argument("-dropout", "--dropout", type=float, help="dropout")
    parser.add_argument("-avg", "--avg", action="store_true",  help="If averaging ")
    parser.add_argument("-mlp", "--mlp", action="store_true",  help="If MLP model")
    parser.add_argument("-bilstm", "--bilstm", action="store_true",  help="If bilstm model")
    parser.add_argument("-original", "--original", action="store_true",  help="If original results were better ")
    parser.add_argument("-layers", "--layers", type=int,  help="Number of layers bilstm ")
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


def process_data(X_train, X_dev, X_test):
    X_train = [" ".join(word_tokenize(item)) for item in X_train]
    X_dev = [" ".join(word_tokenize(item)) for item in X_dev]
    X_test = [" ".join(word_tokenize(item)) for item in X_test]
    # prepare tokenizer
    t = Tokenizer()
    t.fit_on_texts(X_train)
    vocab_size = len(t.word_index) + 1
    # integer encode the documents
    encoded_docs = t.texts_to_sequences(X_train)
    #print(encoded_docs)

    max_length = 300
    padded_docs = pad_sequences(encoded_docs, maxlen=max_length, padding='post')
    #print(padded_docs)

    X_test_encoded = t.texts_to_sequences(X_test)
    X_test_padded = pad_sequences(X_test_encoded, maxlen=max_length, padding="post")

    X_dev_encoded = t.texts_to_sequences(X_dev)
    X_dev_padded = pad_sequences(X_dev_encoded, maxlen=max_length, padding="post")

    return t, vocab_size, padded_docs, encoded_docs, max_length, X_dev_padded, X_test_padded

def run_final_lstm_model(vocab_size, t, embeddings_index, input_length, dim, padded_docs, labels, X_dev, y_dev, X_test, y_test):
    # TODO: add args voor parameters, change args.name to something short
    le = LabelEncoder()
    le.fit(labels)
    labels = le.transform(labels)
    labels = np_utils.to_categorical(labels)
    y_test = le.transform(y_test)
    y_test = np_utils.to_categorical(y_test)
    y_dev = le.transform(y_dev)
    y_dev = np_utils.to_categorical(y_dev)
    print("ytest", y_test)

    embedding_matrix = zeros((vocab_size, dim))
    for word, i in t.word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = np.array(embedding_vector, dtype="float32")
    # define model
    dev_predictions = []
    test_predictions = []
    with open("results_final_lstm_{0}.txt".format(args.name), "w", encoding="utf-8") as outfile:
        if args.avg:
            for i in range(1,11):
                if args.original:  
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
                    h= model.fit(padded_docs, labels, batch_size=16, epochs=50, verbose=0)
                    # evaluate the model
                    loss, accuracy = model.evaluate(X_dev, y_dev, verbose=0)
                    dev_preds = model.predict(X_dev, batch_size=args.batch_size, verbose=0)
                    dev_predictions.append(dev_preds)
                    np.save("lstm_model_{0}_dev_preds_run_{1}.npy".format(args.name, i), dev_preds)
                    test_preds = model.predict(X_test,batch_size=args.batch_size, verbose=0)
                    print("test preds", test_preds)
                    test_predictions.append(test_preds)
                    np.save("lstm_model_{0}_test_preds_run_{1}.npy".format(args.name, i), test_preds)
                    model_json = model.to_json()
                    with open("lstm_model_{0}_run_{1}.json".format(args.name, i), "w") as json_file:
                        json_file.write(model_json)
                    # serialize weights to HDF5
                    model.save_weights("lstm_model_{0}_run_{1}_weights.h5".format(args.name, i))
                    print("Saved model to disk")
                    print("Accuracy run {0}: {1}\n".format(i, accuracy))
                    outfile.write("Accuracy run {0}: {1}\n".format(i, accuracy))
                elif args.bilstm:
                    model = Sequential()
                    model.add(Embedding(vocab_size, dim, weights=[embedding_matrix], input_length=input_length, trainable=False))
                    if args.layers == 1:
                        model.add(Bidirectional(LSTM(int(64), activation="tanh")))
                        model.add(Dropout(float(0.3)))
                    elif args.layers == 2:
                        model.add(Bidirectional(LSTM(int(64), return_sequences=True, activation="tanh")))
                        model.add(Dropout(float(0.3)))
                        model.add(Bidirectional(LSTM(int(64), activation="tanh")))
                        model.add(Dropout(float(0.3)))
                    if args.gender:
                        model.add(Dense(2, activation='softmax'))
                        ## or 
                        #model.add(TimeDistributed(Dense(vocab_size), activation="softmax"))
                    else:
                        model.add(Dense(9, activation='softmax'))
                        ## or
                        #model.add(TimeDistributed(Dense(vocab_size), activation="softmax"))
                    model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=["acc"])
                    h = model.fit(padded_docs, labels, epochs=30, batch_size=16)
                    # evaluate the model
                    loss, accuracy = model.evaluate(X_dev, y_dev, verbose=0)
                    dev_preds = model.predict(X_dev, batch_size=16, verbose=0)
                    dev_predictions.append(dev_preds)
                    np.save("bilstm_model_{0}_dev_preds_run_{1}.npy".format(args.name, i), dev_preds)
                    test_preds = model.predict(X_test,batch_size=16, verbose=0)
                    print("test preds", test_preds)
                    test_predictions.append(test_preds)
                    np.save("bilstm_model_{0}_test_preds_run_{1}.npy".format(args.name, i), test_preds)
                    model_json = model.to_json()
                    with open("bilstm_model_{0}_run_{1}.json".format(args.name, i), "w") as json_file:
                        json_file.write(model_json)
                    # serialize weights to HDF5
                    model.save_weights("bilstm_model_{0}_run_{1}_weights.h5".format(args.name, i))
                    print("Saved model to disk")
                    print("Accuracy run {0}: {1}\n".format(i, accuracy))
                    outfile.write("Accuracy run {0}: {1}\n".format(i, accuracy))
                elif args.mlp:
                    model = Sequential()
                    e = Embedding(vocab_size, dim, weights=[embedding_matrix], input_length=input_length, trainable=False)
                    model.add(e)
                    #model.add(Flatten())
                    model.add(Dense(64, activation='relu'))
                    model.add(Dropout(0.1))
                    model.add(Dense(64, activation='relu'))
                    model.add(Dropout(0.1))
                    model.add(Flatten())
                    if args.gender:
                        model.add(Dense(2, activation='softmax'))
                    else:
                        model.add(Dense(9, activation='softmax'))
                    # compile the model
                    model.compile(optimizer="adam", loss='categorical_crossentropy', metrics=['acc'])
                    # summarize the model
                    print(model.summary())
                    # fit the model
                    model.fit(padded_docs, labels, batch_size=32, epochs=15, verbose=0)
                    # evaluate the model
                    loss, accuracy = model.evaluate(X_dev, y_dev, verbose=0)
                    dev_preds = model.predict(X_dev, batch_size=args.batch_size, verbose=0)
                    dev_predictions.append(dev_preds)
                    np.save("mlp_model_{0}_dev_preds_run_{1}.npy".format(args.name, i), dev_preds)
                    test_preds = model.predict(X_test,batch_size=args.batch_size, verbose=0)
                    print("test preds", test_preds)
                    test_predictions.append(test_preds)
                    np.save("mlp_model_{0}_test_preds_run_{1}.npy".format(args.name, i), test_preds)
                    model_json = model.to_json()
                    with open("mlp_model_{0}_run_{1}.json".format(args.name, i), "w") as json_file:
                        json_file.write(model_json)
                    # serialize weights to HDF5
                    model.save_weights("mlp_model_{0}_run_{1}_weights.h5".format(args.name, i))
                    print("Saved model to disk")
                    print("Accuracy run {0}: {1}\n".format(i, accuracy))
                    outfile.write("Accuracy run {0}: {1}\n".format(i, accuracy))
                else:
                    model = Sequential()
                    e = Embedding(vocab_size, dim, weights=[embedding_matrix], input_length=input_length, trainable=False)
                    model.add(e)
                    model.add(LSTM(args.neurons, return_sequences=True))
                    model.add(LSTM(args.neurons, return_sequences=True))  # returns a sequence of vectors of dimension 32
                    model.add(LSTM(args.neurons))  # return a single vector of dimension 32
                    model.add(Dropout(args.dropout))
                    if args.gender:
                        model.add(Dense(2, activation='softmax'))
                        ## or 
                        #model.add(TimeDistributed(Dense(vocab_size), activation="softmax"))
                    else:
                        model.add(Dense(9, activation='softmax'))
                        ## or
                        #model.add(TimeDistributed(Dense(vocab_size), activation="softmax"))
                    # compile the model
                    model.compile(optimizer=args.opt, loss='categorical_crossentropy', metrics=['acc'])
                    # summarize the model
                    print(model.summary())
                    # fit the model
                    h = model.fit(padded_docs, labels, batch_size=args.batch_size, epochs=args.epochs, verbose=0)
                    # evaluate the model
                    loss, accuracy = model.evaluate(X_dev, y_dev, verbose=0)
                    dev_preds = model.predict(X_dev, batch_size=args.batch_size, verbose=0)
                    dev_predictions.append(dev_preds)
                    np.save("lstm_model_{0}_dev_preds_run_{1}.npy".format(args.name, i), dev_preds)
                    test_preds = model.predict(X_test,batch_size=args.batch_size, verbose=0)
                    print("test preds", test_preds)
                    test_predictions.append(test_preds)
                    np.save("lstm_model_{0}_test_preds_run_{1}.npy".format(args.name, i), test_preds)
                    model_json = model.to_json()
                    with open("lstm_model_{0}_run_{1}.json".format(args.name, i), "w") as json_file:
                        json_file.write(model_json)
                    # serialize weights to HDF5
                    model.save_weights("lstm_model_{0}_run_{1}_weights.h5".format(args.name, i))
                    print("Saved model to disk")
                    print("Accuracy run {0}: {1}\n".format(i, accuracy))
                    outfile.write("Accuracy run {0}: {1}\n".format(i, accuracy))

            averaged_dev_predictions = [list(np.mean(item, axis=0)) for item in list(zip(*dev_predictions))]
            averaged_test_predictions = [list(np.mean(item, axis=0)) for item in list(zip(*test_predictions))]
            np.save("lstm_model_{0}_averaged_dev_predictions_runs1_10.npy".format(args.name), averaged_dev_predictions)
            np.save("lstm_model_{0}_averaged_test_predictions_runs1_10.npy".format(args.name), averaged_test_predictions)
            print("averaged accuracy", accuracy_score(np.argmax(y_test, axis=1), np.argmax(averaged_test_predictions, axis=1)))
        else:
            model = Sequential()
            e = Embedding(vocab_size, dim, weights=[embedding_matrix], input_length=input_length, trainable=False)
            model.add(e)
            model.add(LSTM(args.neurons, return_sequences=True))
            model.add(LSTM(args.neurons, return_sequences=True))  # returns a sequence of vectors of dimension 32
            model.add(LSTM(args.neurons))  # return a single vector of dimension 32
            model.add(Dropout(args.dropout))
            if args.gender:
                model.add(Dense(2, activation='softmax'))
                ## or 
                #model.add(TimeDistributed(Dense(vocab_size), activation="softmax"))
            else:
                model.add(Dense(9, activation='softmax'))
                ## or
                #model.add(TimeDistributed(Dense(vocab_size), activation="softmax"))
            # compile the model
            model.compile(optimizer=args.opt, loss='categorical_crossentropy', metrics=['acc'])
            # summarize the model
            print(model.summary())
            # fit the model
            h = model.fit(padded_docs, labels, batch_size=args.batch_size, epochs=args.epochs, verbose=0)
            # evaluate the model
            loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
            print("Accuracy: {0}, with batch_size {1}, epochs {2}, optimizer {3}, dropout {4}, neurons {5}\n".format(accuracy, args.batch_size, args.epochs, args.opt, args.dropout, args.neurons))
            outfile.write("Accuracy: {0}, with batch_size {1}, epochs {2}, optimizer {3}, dropout {4}, neurons {5}\n".format(accuracy, args.batch_size, args.epochs, args.opt, args.dropout, args.neurons))
    # name should have the description of the task and dimensions

def run_mlp_model(vocab_size, t, embeddings_index, input_length, dim, padded_docs, labels, X_test, y_test):
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

    batch_sizes = [16, 32]
    epoch_size = [15,30]
    dropout_sizes = [0.1, 0.3]
    optimizers = ["adam", "adagrad"]
    neuron_sizes = [64, 128]
    batch_column = []
    epoch_column = []
    dropout_column = []
    optimizer_column = []
    accuracy_column = []
    neurons_column = []


    # define model
    with open("results_file_mlp_{0}.txt".format(args.name), "w", encoding="utf-8") as outfile:
        for batch_size in batch_sizes:
            for drop in dropout_sizes:
                for opt in optimizers:
                    for neurons in neuron_sizes:
                        for ep in epoch_size:
                            model = Sequential()
                            e = Embedding(vocab_size, dim, weights=[embedding_matrix], input_length=input_length, trainable=False)
                            model.add(e)
                            #model.add(Flatten())
                            model.add(Dense(neurons, activation='relu'))
                            model.add(Dropout(drop))
                            model.add(Dense(neurons, activation='relu'))
                            model.add(Dropout(drop))
                            model.add(Flatten())
                            if args.gender:
                                model.add(Dense(2, activation='softmax'))
                            else:
                                model.add(Dense(9, activation='softmax'))
                            # compile the model
                            model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['acc'])
                            # summarize the model
                            print(model.summary())
                            # fit the model
                            model.fit(padded_docs, labels, batch_size=batch_size, epochs=ep, verbose=0)
                            #weights = model.layers[0].get_weights()[0]

                            # # evaluate the model
                            loss, accuracy = model.evaluate(X_test, y_test, verbose=0)

                            batch_column.append(batch_size)
                            epoch_column.append(ep)
                            dropout_column.append(drop)
                            optimizer_column.append(opt)
                            neurons_column.append(neurons)
                            accuracy_column.append(accuracy)
                            print("Accuracy: {0}, with batch_size {1}, optimizer {2}, epochs {3}, neurons {4}, dropout {5}\n".format(accuracy, batch_size, opt, ep, neurons, drop))
                            outfile.write("Accuracy: {0}, with batch_size {1}, optimizer {2}, epochs {3}, neurons {4}, dropout {5}\n".format(accuracy, batch_size, opt, ep, neurons, drop))

    df = pd.DataFrame({"neurons": neurons_column, "batch_size": batch_column, "epochs": epoch_column, "dropout": dropout_column, "optimizer": optimizer_column, "accuracy": accuracy_column})
    df.to_csv("results_mlp_{0}.csv".format(args.name), index=False)
    

def run_BILSTM_model(vocab_size, t, embeddings_index, input_length, dim, padded_docs, labels, X_test, y_test):
    """ X = training instances, y = labels, max_length = max length of a text (padded if shorter, shortened if longer), name is the name of the model, the other arguments are parameters to the NN model 
    This function prepares the data, trains the model, makes predictions, and this is iterated 10 times. The average of the 10 iterations is returned. """
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

    no_layers = [1,2]
    batch_column = []
    epoch_column = []
    dropout_column = []
    optimizer_column = []
    accuracy_column = []
    neurons_column = []
    layers_column = []
    
    ## TO DO: once you've found best options, run model 10 times and get average, plus maybe ensemble?
    with open("results_file_bilstm_{0}.txt".format(args.name), "w", encoding="utf-8") as outfile:
        for layer in no_layers:
            model = Sequential()
            model.add(Embedding(vocab_size, dim, weights=[embedding_matrix], input_length=input_length, trainable=False))
            if layer == 1:
                model.add(Bidirectional(LSTM(int(args.neurons), activation="tanh")))
                model.add(Dropout(float(args.dropout)))
            elif layer == 2:
                model.add(Bidirectional(LSTM(int(args.neurons), return_sequences=True, activation="tanh")))
                model.add(Dropout(float(args.dropout)))
                model.add(Bidirectional(LSTM(int(args.neurons), activation="tanh")))
                model.add(Dropout(float(args.dropout)))
            if args.gender:
                model.add(Dense(2, activation='softmax'))
                ## or 
                #model.add(TimeDistributed(Dense(vocab_size), activation="softmax"))
            else:
                model.add(Dense(9, activation='softmax'))
                ## or
                #model.add(TimeDistributed(Dense(vocab_size), activation="softmax"))
            model.compile(loss='categorical_crossentropy', optimizer=args.opt, metrics=["acc"])
            h = model.fit(padded_docs, labels, epochs=int(args.epochs), batch_size=int(args.batch_size))

            # # evaluate the model
            loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
            print("Accuracy: {0}, with batch_size {1}, optimizer {2}, epochs {3}, neurons {4}, dropout {5}, layers {6}\n".format(accuracy, args.batch_size, args.opt, args.epochs, args.neurons, args.dropout, layer))
            outfile.write("Accuracy: {0}, with batch_size {1}, optimizer {2}, epochs {3}, neurons {4}, dropout {5}, layers {6}\n".format(accuracy, args.batch_size, args.opt, args.epochs, args.neurons, args.dropout, layer))
            batch_column.append(int(args.batch_size))
            epoch_column.append(int(args.epochs))
            dropout_column.append(float(args.dropout))
            optimizer_column.append(args.opt)
            neurons_column.append(int(args.neurons))
            accuracy_column.append(accuracy)
            layers_column.append(layer)
            
    df = pd.DataFrame({"neurons": neurons_column, "batch_size": batch_column, "epochs": epoch_column, "dropout": dropout_column, "optimizer": optimizer_column, "layers": layers_column, "accuracy": accuracy_column})
    df.to_csv("results_bilstm_{0}.csv".format(args.name), index=False)


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

    batch_sizes = [16, 32]
    epoch_size = [15, 30]
    dropout_sizes = [0.1, 0.3]
    optimizers = ["adam", "adagrad"]
    neuron_sizes = [64, 128]

    embedding_matrix = zeros((vocab_size, dim))
    for word, i in t.word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = np.array(embedding_vector, dtype="float32")
    # define model
    batch_column = []
    epoch_column = []
    dropout_column = []
    optimizer_column = []
    accuracy_column = []
    neurons_column = []
    with open("results_file_lstm_{0}.txt".format(args.name), "w", encoding="utf-8") as outfile:
        for batch_size in batch_sizes:
            for dropout_size in dropout_sizes:
                for ep in epoch_size:
                    for opt in optimizers:
                        for neurons in neuron_sizes:
                            model = Sequential()
                            e = Embedding(vocab_size, dim, weights=[embedding_matrix], input_length=input_length, trainable=False)
                            model.add(e)
                            model.add(LSTM(neurons, return_sequences=True))
                            model.add(LSTM(neurons, return_sequences=True))  # returns a sequence of vectors of dimension 32
                            model.add(LSTM(neurons))  # return a single vector of dimension 32
                            model.add(Dropout(dropout_size))
                            if args.gender:
                                model.add(Dense(2, activation='softmax'))
                                ## or 
                                #model.add(TimeDistributed(Dense(vocab_size), activation="softmax"))
                            else:
                                model.add(Dense(9, activation='softmax'))
                                ## or
                                #model.add(TimeDistributed(Dense(vocab_size), activation="softmax"))
                            # compile the model
                            model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['acc'])
                            # summarize the model
                            print(model.summary())
                            # fit the model
                            h = model.fit(padded_docs, labels, batch_size=batch_size, epochs=ep, verbose=0)
                            # evaluate the model
                            loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
                            print("Accuracy: {0}, with batch_size {1}, epochs {2}, optimizer {3}, dropout {4}, neurons {5}\n".format(accuracy, batch_size, ep, opt, dropout_size, neurons))
                            outfile.write("Accuracy: {0}, with batch_size {1}, epochs {2}, optimizer {3}, dropout {4}, neurons {5}\n".format(accuracy, batch_size, ep, opt, dropout_size, neurons))
                            batch_column.append(batch_size)
                            epoch_column.append(ep)
                            dropout_column.append(dropout_size)
                            optimizer_column.append(opt)
                            neurons_column.append(neurons)
                            accuracy_column.append(accuracy)
    # name should have the description of the task and dimensions

    df = pd.DataFrame({"neurons": neurons_column, "batch_size": batch_column, "epochs": epoch_column, "dropout": dropout_column, "optimizer": optimizer_column, "accuracy": accuracy_column})
    df.to_csv("results_lstm_{0}.csv".format(args.name), index=False)

if __name__ == "__main__":
    args = create_arg_parser()
    ## 0. Process data
    X_train, y_train, X_dev, y_dev, X_test, y_test = load_data(args.data_file, header_present=[0])
    t, vocab_size, padded_docs, encoded_docs, max_length, X_dev, X_test = process_data(X_train, X_dev, X_test)
    dim = int(args.dim)
    with open(args.json_emb, "r", encoding="utf-8") as json_file:
        embeddings_index = json.load(json_file)
        print(len(embeddings_index))

    if args.type.lower() == "lstm":
        # print("running lstm...")
        # run_lstm_model(vocab_size, t, embeddings_index, max_length, dim, padded_docs, y_train, X_dev, y_dev)
        #print("running bilstm...")
        #run_BILSTM_model(vocab_size, t, embeddings_index, max_length, dim, padded_docs, y_train, X_dev, y_dev)
        print("running final lstm..")
        run_final_lstm_model(vocab_size, t, embeddings_index, max_length, dim, padded_docs, y_train, X_dev, y_dev, X_test, y_test)
    elif args.type.lower() == "mlp":
        # print("running mlp...")
        # # TO DO MLP
        # run_mlp_model(vocab_size, t, embeddings_index, max_length, dim, padded_docs, y_train, X_dev, y_dev)
        # print("running lstm...")
        # run_lstm_model(vocab_size, t, embeddings_index, max_length, dim, padded_docs, y_train, X_dev, y_dev)
        #print("running bilstm...")
        #run_BILSTM_model(vocab_size, t, embeddings_index, max_length, dim, padded_docs, y_train, X_dev, y_dev)
        print("running final lstm..")
        run_final_lstm_model(vocab_size, t, embeddings_index, max_length, dim, padded_docs, y_train, X_dev, y_dev, X_test, y_test)
