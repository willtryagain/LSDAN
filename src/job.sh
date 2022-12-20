#!/bin/bash
#SBATCH -A research
#SBATCH -n 40
#SBATCH --gres=gpu:4
#SBATCH --mem-per-cpu=2048
#SBATCH --output=op_file.txt
#SBATCH --time=4-00:00:00
#SBATCH --mail-type=END
module load cuda/10.0
module add cuda/10.0


export CUDA_VISIBLE_DEVICES=0,1,2,3

cd /scratch/
if [ ! -d aman.atman ]; then
    mkdir aman.atman
fi

cd aman.atman/
if [ ! -d LSDAN ]; then
    mkdir LSDAN
fi

cd LSDAN/
# rm -r *
rsync -avz aman.atman@ada.iiit.ac.in:/home2/aman.atman/LSDAN/ --exclude=runs --exclude=src/__pycache/  --exclude=src/runs/ ./

# Activate the conda environment
eval "$(conda shell.bash hook)"
conda activate g

cd src
ls
python run.py


