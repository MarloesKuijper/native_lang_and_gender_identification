#!/bin/bash
#SBATCH --time=4:30:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2
#SBATCH --mem=16000
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=marloes.madelon@gmail.com
module load Python/3.6.4-foss-2018a
python get_out_of_vocab_words.py -infile ../my_data2/data_final_cross_genre_native_lang_tm.csv -m twitter