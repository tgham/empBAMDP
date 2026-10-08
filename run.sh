#!/bin/bash -l

#SBATCH -J empBAMDP
#SBATCH -D /home/tgraham/empBAMDP/

#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --partition=compute
#SBATCH --mail-type=ALL
#SBATCH --mail-user=thomas.graham@tuebingen.mpg.de

micromamba activate chickpeas


#### sweep over alpha and ell

# srun python -u ell_alpha_sweep.py --n_ell_samples 1000 --alphas 0.01 0.1 0.25 0.4 1.0 --horizons 1 2 3 --n_arms 3 --n_trials 10 --n_outcomes 4 --costs 0 --init_t 2


#### empirical performance of agents

# srun python -u emp_scoring.py --n_trials 8 --n_outcomes 4 --n_arms 3 --n_rooms 30 --alpha 0.4 --init_t 1 --n_sims 25000 --greedy --termination_arm


#### sweep for diagnostic trials

### arms expt
# srun python -u diag_sweep.py --target ell --n_trials 10  --n_arms 3 --alphas 0.01 0.1 0.25 0.4 1.0  --horizons 1 2 3 --n_ell_samples 2000 --expt arms

### rooms expt
srun python -u diag_sweep.py --target model --n_trials 10  --n_arms 3 --alphas 0.01 0.1 0.25 0.4 1.0 --n_ell_samples 2000 --ell_prior truncnorm --expt rooms 


#### parameter recovery

### arms, no preset histories
# srun python -u emp_recovery.py --n_trials 8 --n_outcomes 4 --n_arms 3 --n_rooms 80 --alpha 0.25 --init_t 1 --n_sims 5000 --ell_bounds 0.01 10 --temp_bounds 0.01 0.3 --gen_data --agent_types emp

### arms, preset histories
# srun python -u emp_recovery.py --n_trials 10 --n_outcomes 4 --n_arms 3 --cost 0.03125  --n_rooms 100 --alpha 0.25 --init_t 0 --termination_arm --n_sims 5000 --ell_bounds 0.01 10 --temp_bounds 0.001 0.1 --gen_data --preset_histories --diag_target model 

### rooms
srun python -u emp_recovery.py --n_trials 10 --n_outcomes 4 --n_arms 3 --cost 0.0 --n_rooms 100 --alpha 0.25 --init_t 0  --n_sims 1000 --ell_bounds 0.01 10 --temp_bounds 0.001 0.1 --gen_data --preset_histories --diag_target model --diag_cols mi_emp mi_model --expt rooms --ell_prior truncnorm


#### fit behavioural data
# srun python -u fit_model.py --n_arms 3 --n_outcomes 4 --n_trials 8 --cost 0.11111 --termination_arm --alpha 0.4 --horizon 3  --init_t 1 
