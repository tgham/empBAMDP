#!/bin/bash -l

#SBATCH -J empBAMDP
#SBATCH -D /home/tgraham/empBAMDP/

#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --partition=compute
#SBATCH --mail-type=ALL
#SBATCH --mail-user=thomas.graham@tuebingen.mpg.de


# micromamba activate chickpeas
# srun python -u ell_alpha_sweep.py --n_ell_samples 1000 --alphas 0.0125 0.025 0.05 0.1 0.2 0.4 0.7 1.0 --horizons 1 8 --n_trials 8 --n_outcomes 4 --termination_arm --ks 0 --skip_t0
# srun python -u ell_alpha_sweep.py --n_ell_samples 1000 --alphas 0.25 1 --horizons 1 8 --n_trials 8 --n_outcomes 4 --termination_arm --ks 0 --contexts 0.25 1 --independent_contexts --init_t 2
# srun python -u ell_alpha_sweep.py --n_ell_samples 1000 --alphas 0.4 --horizons 1 8 --n_arms 3 --n_trials 8 --n_outcomes 4 --termination_arm --ks 0 --init_t 2

# micromamba activate chickpeas
# srun python -u emp_recovery.py --n_trials 8 --n_outcomes 4 --n_arms 3 --n_rooms 30 --alpha 0.4 --init_t 1 --n_sims 5000 --termination_arm --gen_data

micromamba activate chickpeas
srun python -u emp_scoring.py --n_trials 8 --n_outcomes 4 --n_arms 3 --n_rooms 30 --alpha 0.4 --init_t 1 --n_sims 5000 --greedy