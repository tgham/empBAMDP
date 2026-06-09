#!/bin/bash -l

#SBATCH -J empBAMDP
#SBATCH -D /home/tgraham/empBAMDP/

#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --partition=compute
#SBATCH --mail-type=ALL
#SBATCH --mail-user=thomas.graham@tuebingen.mpg.de


micromamba activate chickpeas
# srun python -u ell_alpha_sweep.py --n_ell_samples 1000 --alphas 0.0125 0.025 0.05 0.1 0.2 0.4 0.7 1.0 --n_trials 6 --n_outcomes 4 --termination_arm --ks 0 
srun python -u ell_alpha_sweep.py --n_ell_samples 1000 --alphas 0.1 1 --n_trials 6 --n_outcomes 4 --termination_arm --ks 0 --contexts 0.1 1




