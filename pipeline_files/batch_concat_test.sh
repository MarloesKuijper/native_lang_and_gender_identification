#!/bin/bash
#SBATCH --time=2:30:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32000
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=marloes.madelon@gmail.com
module load Python/3.6.4-foss-2018a

python concat_embeddings.py -emb1 ../word_embedding_files/glove_regular_embeddings_27B25D.json -emb2 ../word_embedding_files/learner_english_embeddings_25d.json -out_reg "../word_embedding_files/concatenated_embeddings_25d_regular.json" -out_pca ../word_embedding_files/concatenated_embeddings_25d_pca.json -dim 25
python concat_embeddings.py -emb1 ../word_embedding_files/glove_regular_embeddings_27B100D.json -emb2 ../word_embedding_files/learner_english_embeddings_100d.json -out_reg "../word_embedding_files/concatenated_embeddings_100d_regular.json" -out_pca ../word_embedding_files/concatenated_embeddings_100d_pca.json -dim 100