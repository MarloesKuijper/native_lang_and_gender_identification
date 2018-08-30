#!/bin/bash
#SBATCH --time=0:10:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=4000
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=marloes.madelon@gmail.com
module load Python/3.6.4-foss-2018a

python run_this.py