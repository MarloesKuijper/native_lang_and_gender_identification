import json
from collections import defaultdict
import argparse
import numpy as np
from sklearn.decomposition import PCA

def create_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-emb1","--emb1", type=str, help="First embeddings (regular json)")
    parser.add_argument("-emb2","--emb2", type=str, help="Second embeddings (learner json)")
    parser.add_argument("-dim","--dim", type=str, help="Dimensions embeddings")
    parser.add_argument("-out_pca","--out_pca", type=str, help="Name concatenated embeddings")
    parser.add_argument("-out_reg","--out_reg", type=str, help="Name concatenated embeddings")
    args = parser.parse_args()
    return args

def create_regular_concat(embeddings_index1, embeddings_index2, dim):
    new_embedding_index = defaultdict(list)

    for item in embeddings_index2.keys():
        if item in embeddings_index1.keys():
            new_embedding_index[item] = embeddings_index2[item] + embeddings_index1[item]

    print("initial length", len(new_embedding_index))
    embedding_matrix_regular = np.array([np.array(item, dtype="float32") for item in embeddings_index1.values() if len(item) == dim])
    embedding_matrix_learner = np.array([np.array(item, dtype="float32") for item in embeddings_index2.values() if len(item) == dim])
    embedding_matrix_reg_mean = np.mean(embedding_matrix_regular, axis=0)
    embedding_matrix_learner_mean = np.mean(embedding_matrix_learner, axis=0)
    embedding_matrix_reg_mean_str = embedding_matrix_reg_mean.astype(str)
    embedding_matrix_learner_mean_str = embedding_matrix_learner_mean.astype(str)
    print("string")
    print(embedding_matrix_learner_mean_str)
    print(embedding_matrix_reg_mean_str)

    print(embedding_matrix_regular[0])
    faulty_items = []
    print("starting first part")
    print(new_embedding_index["twitter"])

    new_embedding_index_key_set = set(list(new_embedding_index.keys()))
    regular_missing_emb_set = set(list(embeddings_index1.keys())).difference(new_embedding_index_key_set)
    learner_missing_emb_set = set(list(embeddings_index2.keys())).difference(new_embedding_index_key_set)

    print("twitter" in regular_missing_emb_set)
    print("twitter" in learner_missing_emb_set)

    print(embedding_matrix_reg_mean)

    for item in regular_missing_emb_set:
        new_embedding_index[item] = list(embedding_matrix_learner_mean_str) + embeddings_index1[item]

    print("length after adding regular", len(new_embedding_index))

    for item in learner_missing_emb_set:   
        new_embedding_index[item] = embeddings_index2[item] + list(embedding_matrix_reg_mean_str)

    print("length after adding learner", len(new_embedding_index))

    with open(args.out_reg, "w") as outfile:
        json.dump(new_embedding_index, outfile)

def create_pca_concat(embeddings_index1, embeddings_index2,dim):
    new_embedding_index = defaultdict(list)

    for item in embeddings_index2.keys():
        if item in embeddings_index1.keys():
            new_embedding_index[item] = embeddings_index2[item] + embeddings_index1[item]
    print(list(new_embedding_index.keys())[:5])
    print("initial length", len(new_embedding_index))

    embedding_matrix = np.array([[float(i) for i in item] for item in new_embedding_index.values()])
    print(embedding_matrix.shape)
    print(embedding_matrix[:5])
    #print("emb matrix", embedding_matrix)
    pca = PCA(n_components=dim)
    embedding_matrix_pca = pca.fit_transform(embedding_matrix)
    #print("reduced emb matrix", embedding_matrix_pca)
    embeddings_dict_after_pca = {item: list(embedding_matrix_pca[ix].astype(str)) for ix, item in enumerate(new_embedding_index.keys())}
    print(list(new_embedding_index.keys())[:5])
    #print(embeddings_dict_after_pca)
    print(embeddings_dict_after_pca["twitter"])

    embedding_matrix_regular = np.array([np.array(item, dtype="float32") for item in embeddings_index1.values() if len(item) == dim])
    embedding_matrix_learner = np.array([np.array(item, dtype="float32") for item in embeddings_index2.values() if len(item) == dim])
    embedding_matrix_reg_mean = np.mean(embedding_matrix_regular, axis=0)
    embedding_matrix_learner_mean = np.mean(embedding_matrix_learner, axis=0)
    embedding_matrix_reg_mean_str = embedding_matrix_reg_mean.astype(str)
    embedding_matrix_learner_mean_str = embedding_matrix_learner_mean.astype(str)

    new_embedding_index_key_set = set(list(embeddings_dict_after_pca.keys()))
    regular_missing_emb_set = set(list(embeddings_index1.keys())).difference(new_embedding_index_key_set)
    learner_missing_emb_set = set(list(embeddings_index2.keys())).difference(new_embedding_index_key_set)
    print(len(regular_missing_emb_set), len(learner_missing_emb_set))

    for item in regular_missing_emb_set:
        embeddings_dict_after_pca[item] = embeddings_index1[item]

    print("length after adding regular", len(embeddings_dict_after_pca))

    for item in learner_missing_emb_set:   
        embeddings_dict_after_pca[item] = embeddings_index2[item]
    print("length after adding learner", len(embeddings_dict_after_pca))

    with open(args.out_pca, "w") as outfile:
        json.dump(embeddings_dict_after_pca, outfile)


if __name__ == "__main__":
    args = create_arg_parser()
    with open(args.emb1, "r", encoding="utf-8") as json_file_r, open(args.emb2, "r", encoding="utf-8") as json_file_l:
            embeddings_index_r = json.load(json_file_r, strict=False)
            embeddings_index_l = json.load(json_file_l, strict=False)

    #create_regular_concat(embeddings_index_r, embeddings_index_l, int(args.dim))

    create_pca_concat(embeddings_index_r, embeddings_index_l, int(args.dim))

# vocab = 40000
# embeddings = 89000 words
# voor 89000 words krijg je betere embeddings maar dit helpt dus niet met out of vocab (e.g. dus woorden die in vocab zitten maar niet in embeddings)
## so controlling for size does create equal embeddings but it does not really make the two equal because the learners will have an advnatage
## namely, they are trained on myown data so there wont be out of vocab words, whereas with pretrained onl ythe top X words are used, thereby missing out on many words in my data
