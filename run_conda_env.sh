#!/bin/bash

#Informacion del trabajo
#SBATCH --job-name=RajagData
#SBATCH -o results/results_Rajagopalan_Val_Datasets%j.out
#SBATCH -e error/error_Rajagopalan_Val_Datasets%j.err

#Recursos
#SBATCH --partition=cascadelakegpu
#SBATCH --qos=normal
#SBATCH -n 8
#SBATCH --mem=0
#SBATCH --gres=gpu:1

#SBATCH --time=3-00:00:00
#SBATCH --mail-user=sljuarez@ubu.es
#SBATCH --mail-type=ALL

#Directorio de trabajo
#SBATCH -D .

#Cargamos las variables necesarias y el entorno conda
module load cascadelake/CUDA_10.1
export PATH=/home/ubu_eps_1/COMUNES/miniconda3/bin:$PATH
source /home/ubu_eps_1/COMUNES/miniconda3/etc/profile.d/conda.sh
conda activate env

python Rajagopalan_Val_Datasets.py


#Desactivamos el entorno
conda deactivate
