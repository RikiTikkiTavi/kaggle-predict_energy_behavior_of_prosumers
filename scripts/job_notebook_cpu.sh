#!/bin/bash

#SBATCH --job-name="ebop-notebook"
#SBATCH --output="/beegfs/ws/0/s4610340-energy_behavior/dev/logs/job_notebook_cpu/slurm-%j.out"
#SBATCH --error="/beegfs/ws/0/s4610340-energy_behavior/dev/logs/job_notebook_cpu/slurm-%j.out"
#SBATCH --account="p_scads"
#SBATCH --time=8:00:00
#SBATCH --partition=alpha
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=10000

module load GCC/11.3.0 Python/3.10.4 CUDA/12.0.0

source /beegfs/ws/0/s4610340-energy_behavior/yahor/kaggle-predict_energy_behavior_of_prosumers/.venv/bin/activate

export XDG_RUNTIME_DIR=""

cd /beegfs/ws/0/s4610340-energy_behavior/yahor/kaggle-predict_energy_behavior_of_prosumers/notebooks && jupyter notebook --no-browser --port=8888 --NotebookApp.token=fb907dde42aafe7bd38b25a928cfe18225404f8b8ea30f20
