#!/bin/bash -l
#SBATCH -p compute                 #specify partition
#SBATCH -N 1                       #specify number of nodes
#SBATCH --cpus-per-task=1          #specify number of cpus
#SBATCH -t 1:00:00                 #job time limit <hr:min:sec>
#SBATCH -J tgai_tana_005                #job name
#SBATCH -A cb900901                 #specify your account ID


echo "Welcome to LANTA"
echo "use 'myqueue' command to check job status"
python3 trainModel.py
