#!/bin/bash -l

#SBATCH -J empBAMDP
#SBATCH -D /home/tgraham/empBAMDP/

#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --partition=compute
#SBATCH --mail-type=ALL
#SBATCH --mail-user=thomas.graham@tuebingen.mpg.de


# micromamba activate chickpeas
srun python -u ell_alpha_sweep.py --n_ell_samples 1000 --alphas 0.01 0.15 0.3 0.45 0.6 0.75 0.9 1.0 --n_trials 6




