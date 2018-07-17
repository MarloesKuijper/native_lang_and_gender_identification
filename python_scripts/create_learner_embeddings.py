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
    parser.add_argument("-data_file","--data_file", type=str, help="Data file")
    parser.add_argument("-cambridge", "--cambridge", type=str, help="Cambridge sentences")
    parser.add_argument("-wer_txt", "--wer_txt", type=str, help="Pretrained regular embeddings (txt)")
    parser.add_argument("-wel_txt", "--wel_txt", type=str, help="Learner embeddings (txt)")
    parser.add_argument("-wel_json", "--wel_json", type=str, help="Learner embeddings as dict (json)")
    parser.add_argument("-type", "--type", required=True, type=str, help="Pick the type: learner or learner_init")
    parser.add_argument("-dim", "--dim", required=True, type=str, help="Dimensions")
    args = parser.parse_args()
    return args

def load_regular_pretrained_glove_embeddings(pretrained_embeddings, dim):
    word2vec_output_file = 'output.txt.word2vec'
    glove2word2vec(pretrained_embeddings, word2vec_output_file)
    # load the Stanford GloVe model
    filename = 'output.txt.word2vec'
    model = KeyedVectors.load_word2vec_format(filename, binary=False, limit=71054)
    # calculate: (king - man) + woman = ?
    #result = model.most_similar(positive=['woman', 'king'], negative=['man'], topn=1)
    #print(result)
    model.wv.save_word2vec_format(args.wel_txt, binary=False)

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
    # X_train, y_train, X_dev, y_dev, X_test, y_test = load_data(args.data_file, header_present=[0])
    # t, vocab_size, padded_docs, encoded_docs, max_length, X_dev = process_data(X_train, X_dev)
    dim = int(args.dim)
    countries = ["germany", "iran", "italy", "new-delhi", "poland", "portugal", "russia", "spain", "the-netherlands"]
    twitter = get_data("../twitter/", countries, "_messages.txt")
    twitter_2 = get_data("../twitter_additional_data/", countries, "_messages.txt")
    medium = get_data("../medium/", countries, "messages_@")
    medium_2 = get_data("../medium_additional_data/", countries, "messages_@")
    corpus_data = twitter + twitter_2 + medium + medium_2

    if args.type == "learner":
        ## 2. create learner embeddings in txt and json
        model = create_learner_embeddings(args.cambridge, corpus_data, dim)
        # # model = KeyedVectors.load_word2vec_format(args.wel_txt, binary=False)
        # # print(model.wv.vocab)
        embeddings_index = create_embeddings_dict(args.wel_txt, args.wel_json)
    elif args.type == "learner_init":
        ## 2. Run learner embeddings, use same Dimensions and same vocab size if possible (or reduce the vocab size of previous)
        model = create_learner_embeddings(args.cambridge, corpus_data, dim, pretrained=args.wer_txt)
        # # model = KeyedVectors.load_word2vec_format(args.wel_txt, binary=False)
        # # print(model.wv.vocab)
        embeddings_index = create_embeddings_dict(args.wel_txt, args.wel_json)

    elif args.type == "regular_control":
        load_regular_pretrained_glove_embeddings(args.wer_txt, dim)
        embeddings_index = create_embeddings_dict(args.wel_txt, args.wel_json)
