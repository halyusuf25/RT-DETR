#!/bin/bash
#---------------------------------------------------------------
###PART 1: Requests resources to run your job.
# --------------------------------------------------------------
### Optional. Set the job name
#SBATCH --job-name=rtdetr900_studentonly
### Optional. Set the output filename.
### SLURM reads %x as the job name and %j as the job ID
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
### REQUIRED. Specify the PI group for this job
#SBATCH --account=eungjoolee
### Optional. Request email when job begins and ends
#SBATCH --mail-type=ALL
### Optional. Specify email address to use for notification
# SBATCH --mail-user=halyusuf@arizona.edu
### REQUIRED. Set the partition for your job.
#SBATCH --partition=gpu_standard
### REQUIRED. Set the number of cores that will be used for this job.
#SBATCH --ntasks=13
### Optional. Includes GPU
#SBATCH --gres=gpu:volta:2
### REQUIRED. Set the number of nodes
#SBATCH --nodes=1
### REQUIRED. Specify the time required for this job, hhh:mm:ss
#SBATCH --time=95:58:00


# --------------------------------------------------------------
### PART 2: Executes bash commands to run your job
# --------------------------------------------------------------
### Load required modules/libraries if needed
module load python/3.11/3.11.4
module load cuda12/12.4.1 

### Navigate to the root directory containing both 'slurm_jobs' and 'src'
cd /home/u14/halyusuf/github/RT-DETR/

### Add the current directory to PYTHONPATH so that Python finds 'src'
### export PYTHONPATH=$(pwd):$PYTHONPATH

### Activate your environment
source .venv/bin/activate
cd rtdetr_pytorch/

### Run your work
torchrun --nproc_per_node=2 --nnodes=1 tools/train_copy.py -c configs/rtdetr/rtdetr_r50vd_6x_coco.yml
