#!/bin/bash
#SBATCH --time=16:30:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=40000
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=marloes.madelon@gmail.com
module load Python/3.6.4-foss-2018a
python other_algorithms.py -cross -name "results_cross_genre_native_lang_tm_diff_algos" -ff "../my_data2/data_final_cross_genre_native_lang_tm.csv"
python other_algorithms.py -cross -name "results_cross_genre_native_lang_mt_diff_algos" -ff "../my_data2/data_final_cross_genre_native_lang_mt.csv"
python other_algorithms.py -name "results_within_genre_native_lang_twitter_diff_algos" -ff "../my_data2/data_final_within_genre_native_lang_twitter.csv"
python other_algorithms.py -name "results_within_genre_native_lang_medium_diff_algos" -ff "../my_data2/data_final_within_genre_native_lang_medium.csv"
python other_algorithms.py -cross --gender -name "results_cross_genre_gender_tm_diff_algos" -ff "../my_data2/data_final_cross_genre_gender_tm.csv"
python other_algorithms.py -cross --gender -name "results_cross_genre_gender_mt_diff_algos" -ff "../my_data2/data_final_cross_genre_gender_mt.csv"
python other_algorithms.py --gender -name "results_within_genre_gender_twitter_diff_algos" -ff "../my_data2/data_final_within_genre_gender_twitter.csv"
python other_algorithms.py --gender -name "results_within_genre_gender_medium_diff_algos" -ff "../my_data2/data_final_within_genre_gender_medium.csv"