#!/bin/bash
#SBATCH --time=5:30:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2
#SBATCH --mem=16000
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=marloes.madelon@gmail.com
module load Python/3.6.4-foss-2018a

python ../retrofitting/retrofit.py -i ../word_embedding_files/glove.twitter.27B.25d.txt -l "retrofitted_most_informative_feats_native_lang.txt" -n 10 -o ../word_embedding_files/retro_word_embeddings_regular_25d_most_informative.txt
python ../retrofitting/retrofit.py -i ../word_embedding_files/glove.twitter.27B.100d.txt -l "retrofitted_most_informative_feats_native_lang.txt" -n 10 -o ../word_embedding_files/retro_word_embeddings_regular_100d_most_informative.txt

python ../retrofitting/retrofit.py -i ../word_embedding_files/regular_pretrained_embeddings_control_25d.txt -l "retrofitted_most_informative_feats_native_lang.txt" -n 10 -o ../word_embedding_files/retro_word_embeddings_regular_control_25d_most_informative.txt
python ../retrofitting/retrofit.py -i ../word_embedding_files/regular_pretrained_embeddings_control_100d.txt -l "retrofitted_most_informative_feats_native_lang.txt" -n 10 -o ../word_embedding_files/retro_word_embeddings_regular_control_100d_most_informative.txt

python ../retrofitting/retrofit.py -i ../word_embedding_files/learner_english_embeddings_25d.txt -l "retrofitted_most_informative_feats_native_lang.txt" -n 10 -o ../word_embedding_files/retro_word_embeddings_learner_25d_most_informative.txt
python ../retrofitting/retrofit.py -i ../word_embedding_files/learner_english_embeddings_100d.txt -l "retrofitted_most_informative_feats_native_lang.txt" -n 10 -o ../word_embedding_files/retro_word_embeddings_learner_100d_most_informative.txt

python ../retrofitting/retrofit.py -i ../word_embedding_files/glove.twitter.27B.25d.txt -l "retrofitting_file_oov_native_lang.txt" -n 10 -o ../word_embedding_files/retro_word_embeddings_regular_25d_oov.txt
python ../retrofitting/retrofit.py -i ../word_embedding_files/glove.twitter.27B.100d.txt -l "retrofitting_file_oov_native_lang.txt" -n 10 -o ../word_embedding_files/retro_word_embeddings_regular_100d_oov.txt

python ../retrofitting/retrofit.py -i ../word_embedding_files/regular_pretrained_embeddings_control_25d.txt -l "retrofitting_file_oov_native_lang.txt" -n 10 -o ../word_embedding_files/retro_word_embeddings_regular_control_25d_oov.txt
python ../retrofitting/retrofit.py -i ../word_embedding_files/regular_pretrained_embeddings_control_100d.txt -l "retrofitting_file_oov_native_lang.txt" -n 10 -o ../word_embedding_files/retro_word_embeddings_regular_control_100d_oov.txt

python ../retrofitting/retrofit.py -i ../word_embedding_files/learner_english_embeddings_25d.txt -l "retrofitting_file_oov_native_lang.txt" -n 10 -o ../word_embedding_files/retro_word_embeddings_learner_25d_oov.txt
python ../retrofitting/retrofit.py -i ../word_embedding_files/learner_english_embeddings_100d.txt -l "retrofitting_file_oov_native_lang.txt" -n 10 -o ../word_embedding_files/retro_word_embeddings_learner_100d_oov.txt