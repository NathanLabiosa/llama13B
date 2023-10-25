#!/bin/bash
#
##SBATCH --nodes=1
#SBATCH -p gpu
#SBATCH --cpus-per-task=4
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:1
#SBATCH --constraint=gmem48
#SBATCH --job-name=llama_rationales

#SBATCH --error=llama_rationales.%J.err
#SBATCH --output=llama_rationales.%J.out

export job_name="llama_rationales"
export log_dir="/home/nlabiosa/llama13B/llama/job_logs/$job_name-$SLURM_JOB_ID"
mkdir -p $log_dir
export debug_logs="$log_dir/job_$SLURM_JOB_ID.log"
export benchmark_logs="$log_dir/job_$SLURM_JOB_ID.log"

module load cuda/11.0
module load anaconda3/2022.05

cd $SLURM_SUBMIT_DIR

echo "Slurm working directory: $SLURM_SUBMIT_DIR" >> $debug_logs
echo "JobID: $SLURM_JOB_ID" >> $debug_logs
echo "Running on $SLURM_NODELIST" >> $debug_logs
echo "Running on $SLURM_NNODES nodes." >> $debug_logs
echo "Running on $SLURM_NPROCS processors." >> $debug_logs
echo "Current working directory is `pwd`" >> $debug_logs

echo "Modules loaded:" >> $debug_logs
module list >> $debug_logs
echo "mpirun location: $(which mpirun)" >> $debug_logs

echo "Starting time: $(date)" >> $benchmark_logs
echo "ulimit -l: " >> $benchmark_logs
ulimit -l >> $benchmark_logs

### Running the program ###

## Select python File to run
source activate llava
# python e-SNLI_generate_answers.py --model_path "/home/nlabiosa/llama13B/llama/llama-13b-hf" --data_path "/home/nlabiosa/llama13B/llama/e-SNLI_experimentation/esnli_minitest.csv" --output_path "experimentation.jsonl"
#python e-SNLI_evaluate_results.py
python cosine_eval.py
#python blip_generate.py


sleep 3

echo "Ending time: $(date)" >> $benchmark_logs
echo "ulimit -l: " >> $benchmark_logs
ulimit -l >> $benchmark_logs

## Directory Cleanup ##
mv $job_name.$SLURM_JOB_ID.err $log_dir
mv $job_name.$SLURM_JOB_ID.out $log_dir

