#!/bin/bash
#SBATCH --time=4:00:00
#SBATCH --job-name=hmda-parquet
#SBATCH --account=torch_pr_609_general
#SBATCH --mem=16G

#source "/scratch/projects/greenwaldlab/venvs/hmda-second-liens/bin/activate"
#cd "/home/dlg340/research/hmda-second-liens"

python -m py_tools.datasets.hmda convert 2003 --overwrite
