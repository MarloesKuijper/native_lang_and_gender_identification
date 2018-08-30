#!/bin/bash
#SBATCH --time=10:30:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=8000
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=marloes.madelon@gmail.com
module load Python/3.6.4-foss-2018a

## eerste 2 zijn voor learner embeddings, laatste 2 voor initialized learner embeddings met pretrained glove
## ff checken wat hierna de vocab size is van de learner english json (als anders dan eerst, dan moet controlled regualr ook opnieuw)

python create_learner_embeddings.py -wel_txt ../word_embedding_files/learner_english_embeddings_25d.txt -cambridge ../word_embedding_files/sentences.txt -wel_json ../word_embedding_files/learner_english_embeddings_25d.json  -type learner -dim 25 

python create_learner_embeddings.py -wel_txt ../word_embedding_files/learner_english_embeddings_100d.txt -cambridge ../word_embedding_files/sentences.txt -wel_json ../word_embedding_files/learner_english_embeddings_100d.json  -type learner -dim 100 

python create_learner_embeddings.py -wer_txt ../word_embedding_files/glove.twitter.27B.25d.txt -wel_txt ../word_embedding_files/learner_english_embeddings_initialized_25d.txt -cambridge ../word_embedding_files/sentences.txt -wel_json ../word_embedding_files/learner_english_embeddings_initialized_25d.json  -type learner_init -dim 25 

python create_learner_embeddings.py -wer_txt ../word_embedding_files/glove.twitter.27B.100d.txt -wel_txt ../word_embedding_files/learner_english_embeddings_initialized_100d.txt -cambridge ../word_embedding_files/sentences.txt -wel_json ../word_embedding_files/learner_english_embeddings_initialized_100d.json  -type learner_init -dim 100 
