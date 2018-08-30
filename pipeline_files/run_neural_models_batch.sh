#!/bin/bash
#SBATCH --time=10:30:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=8000
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=marloes.madelon@gmail.com
module load Python/3.6.4-foss-2018a

python run_neural_models.py -cross -name "results_cross_genre_native_lang_tm_100d.txt" -data_file "./my_data2/data_final_cross_genre_native_lang_tm.csv" -data_folder . -json_emb "best_embeddings.json" -type lstm -dim 100
python run_neural_models.py -cross -name "results_cross_genre_native_lang_mt_100d.txt" -data_file "./my_data2/data_final_cross_genre_native_lang_mt.csv" -data_folder . -json_emb "best_embeddings.json" -type lstm -dim 100
python run_neural_models.py -name "results_within_genre_native_lang_twitter_100d.txt" -data_file "./my_data2/data_final_within_genre_native_lang_twitter.csv" -data_folder . -json_emb "best_embeddings.json" -type lstm -dim 100
python run_neural_models.py -name "results_within_genre_native_lang_medium_100d.txt" -data_file "./my_data2/data_final_within_genre_native_lang_medium.csv" -data_folder . -json_emb "best_embeddings.json" -type lstm -dim 100
python run_neural_models.py -cross --gender -name "results_cross_genre_gender_tm_100d.txt" -data_file "./my_data2/data_final_cross_genre_gender_tm.csv" -data_folder . -json_emb "best_embeddings.json" -type lstm -dim 100
python run_neural_models.py -cross --gender -name "results_cross_genre_gender_mt_100d.txt" -data_file "./my_data2/data_final_cross_genre_gender_mt.csv" -data_folder . -json_emb "best_embeddings.json" -type lstm -dim 100
python run_neural_models.py --gender -name "results_within_genre_gender_twitter_100d.txt" -data_file "./my_data2/data_final_within_genre_gender_twitter.csv" -data_folder . -json_emb "best_embeddings.json" -type lstm -dim 100
python run_neural_models.py --gender -name "results_within_genre_gender_medium_100d.txt" -data_file "./my_data2/data_final_within_genre_gender_medium.csv" -data_folder . -json_emb "best_embeddings.json" -type lstm -dim 100

python run_neural_models.py -cross -name "results_cross_genre_native_lang_tm_100d.txt" -data_file "./my_data2/data_final_cross_genre_native_lang_tm.csv" -data_folder . -json_emb "best embeddings.json" -type bilstm -dim 100
python run_neural_models.py -cross -name "results_cross_genre_native_lang_mt_100d.txt" -data_file "./my_data2/data_final_cross_genre_native_lang_mt.csv" -data_folder . -json_emb "best embeddings.json" -type bilstm -dim 100
python run_neural_models.py -name "results_within_genre_native_lang_twitter_100d.txt" -data_file "./my_data2/data_final_within_genre_native_lang_twitter.csv" -data_folder . -json_emb "best embeddings.json" -type bilstm -dim 100
python run_neural_models.py -name "results_within_genre_native_lang_medium_100d.txt" -data_file "./my_data2/data_final_within_genre_native_lang_medium.csv" -data_folder . -json_emb "best embeddings.json" -type bilstm -dim 100
python run_neural_models.py -cross --gender -name "results_cross_genre_gender_tm_100d.txt" -data_file "./my_data2/data_final_cross_genre_gender_tm.csv" -data_folder . -json_emb "best embeddings.json" -type bilstm -dim 100
python run_neural_models.py -cross --gender -name "results_cross_genre_gender_mt_100d.txt" -data_file "./my_data2/data_final_cross_genre_gender_mt.csv" -data_folder . -json_emb "best embeddings.json" -type bilstm -dim 100
python run_neural_models.py --gender -name "results_within_genre_gender_twitter_100d.txt" -data_file "./my_data2/data_final_within_genre_gender_twitter.csv" -data_folder . -json_emb "best embeddings.json" -type bilstm -dim 100
python run_neural_models.py --gender -name "results_within_genre_gender_medium_100d.txt"  -data_file "./my_data2/data_final_within_genre_gender_medium.csv" -data_folder . -json_emb "best embeddings.json" -type bilstm -dim 100
