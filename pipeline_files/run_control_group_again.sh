#!/bin/bash
#SBATCH --time=10:30:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=8000
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=marloes.madelon@gmail.com
module load Python/3.6.4-foss-2018a

python create_learner_embeddings.py -wer_txt ../word_embedding_files/glove.twitter.27B.25d.txt -wel_txt ../word_embedding_files/regular_pretrained_embeddings_control_25d.txt -wel_json ../word_embedding_files/regular_pretrained_embeddings_control_25d.json  -type regular_control -dim 25 

python create_learner_embeddings.py -wer_txt ../word_embedding_files/glove.twitter.27B.100d.txt -wel_txt ../word_embedding_files/regular_pretrained_embeddings_control_100d.txt -wel_json ../word_embedding_files/regular_pretrained_embeddings_control_100d.json  -type regular_control -dim 100 
