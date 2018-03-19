#!/usr/bin/env bash
#PBS -l nodes=1:ppn=6
#PBS -l walltime=24:00:00
#PBS -A xzv-031-ab
#PBS -N VPG
#PBS -o ./VPG_out.txt
#PBS -e ./VPG_error.txt
#PBS -q sw

source /software/soft.computecanada.ca.sh
module load python/3.5.2
module scipy-stack/2017b
module load cuda/8.0.44
module load cudnn/7.0

source $RL_ENV

N_REPLICATES=10
CPU_COUNT=1
N_EPISODES=5000
cd $PBS_O_WORKDIR
# runs VPG in some simple domains
python3 vpg.py --env_name Acrobot-v1 --n_episodes $N_EPISODES --n_replicates $N_REPLICATES --cpu_count $CPU_COUNT &
python3 vpg.py --env_name Pendulum-v0 --n_episodes $N_EPISODES --n_replicates $N_REPLICATES --cpu_count $CPU_COUNT &
python3 vpg.py --env_name CartPole-v0 --n_episodes $N_EPISODES --n_replicates $N_REPLICATES --cpu_count $CPU_COUNT &
python3 vpg.py --env_name MountainCar-v0 --n_episodes $N_EPISODES --n_replicates $N_REPLICATES --cpu_count $CPU_COUNT &
python3 vpg.py --env_name MountainCarContinuous-v0 --n_episodes $N_EPISODES --n_replicates $N_REPLICATES --cpu_count $CPU_COUNT &
python3 vpg.py --env_name FrozenLake-v0 --n_episodes $N_EPISODES --n_replicates $N_REPLICATES --cpu_count $CPU_COUNT &
wait