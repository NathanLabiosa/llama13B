#!/bin/bash
#
##SBATCH --nodes=1
#SBATCH -p gpu
#SBATCH --cpus-per-task=4
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:1
#SBATCH --constraint=gmem48
#SBATCH --job-name=llama_dataset_answer_generation

#SBATCH --error=llama_dataset_answer_generation.%J.err
#SBATCH --output=llama_dataset_answer_generation.%J.out

export job_name="llama_dataset_answer_generation"
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
source activate /home/nlabiosa/.conda/envs/llava
#python anli_experimentation/llama_anli_generate_answers.py --model_path "/home/nlabiosa/llama13B/llama/vicuna-7b-v1.1" --data_path "/home/nlabiosa/llama13B/llama/anli/data/anli_v1.0/R1/test.jsonl" --output_path "/home/nlabiosa/llama13B/llama/generated_answers/anli_vicuna7B.jsonl"
#python cos-e_experimentation/cos-e_generate_answers.py --model_path "/home/nlabiosa/llama13B/llama/vicuna-7b-v1.1" --data_path "/home/nlabiosa/llama13B/llama/cos-e/data/v1.0/mini_test.jsonl" --output_path "/home/nlabiosa/llama13B/llama/generated_answers/cos-e_vicuna7B.jsonl"
#python e-SNLI_experimentation/e-SNLI_generate_answers.py --model_path "/home/nlabiosa/llama13B/llama/vicuna-7b-v1.1" --data_path "/home/nlabiosa/llama13B/llama/e-SNLI/dataset/mini_test.csv" --output_path "/home/nlabiosa/llama13B/llama/generated_answers/e-snli_vicuna7B_v2.jsonl"
# python SVAMP_experimentation/SVAMP_generate_answers.py --model_path "/home/nlabiosa/llama13B/llama/vicuna-7b-v1.1" --data_path "/home/nlabiosa/llama13B/llama/SVAMP/data/cv_asdiv-a/fold0/train.csv" --output_path "/home/nlabiosa/llama13B/llama/generated_answers/SVAMP_vicuna7B.jsonl"
# python ScienceQA_experimentation/llama-ScienceQA.py --model_path "/home/nlabiosa/llama13B/llama/vicuna-7b-v1.1" --data_path "/home/nlabiosa/LLaVA/ScienceQA/data/scienceqa/llava_minitest_QCM-LEPA.json" --output_path "/home/nlabiosa/llama13B/llama/generated_answers/ScienceQA_vicuna7B.jsonl"

# python anli_experimentation/llama_anli_generate_answers.py --model_path "/home/nlabiosa/llama13B/llama/vicuna-13b-v1.1" --data_path "/home/nlabiosa/llama13B/llama/anli/data/anli_v1.0/R1/test.jsonl" --output_path "/home/nlabiosa/llama13B/llama/generated_answers/anli_vicuna13B.jsonl"
#python cos-e_experimentation/cos-e_generate_answers.py --model_path "/home/nlabiosa/llama13B/llama/vicuna-13b-v1.1" --data_path "/home/nlabiosa/llama13B/llama/cos-e/data/v1.0/mini_test.jsonl" --output_path "/home/nlabiosa/llama13B/llama/generated_answers/cos-e_vicuna13B.jsonl"
# python e-SNLI_experimentation/e-SNLI_generate_answers.py --model_path "/home/nlabiosa/llama13B/llama/vicuna-13b-v1.1" --data_path "/home/nlabiosa/llama13B/llama/e-SNLI/dataset/mini_test.csv" --output_path "/home/nlabiosa/llama13B/llama/generated_answers/e-snli_vicuna13B_v2.jsonl"
# python SVAMP_experimentation/SVAMP_generate_answers.py --model_path "/home/nlabiosa/llama13B/llama/vicuna-13b-v1.1" --data_path "/home/nlabiosa/llama13B/llama/SVAMP/data/cv_asdiv-a/fold0/train.csv" --output_path "/home/nlabiosa/llama13B/llama/generated_answers/SVAMP_vicuna13B.jsonl"
# python ScienceQA_experimentation/llama-ScienceQA.py --model_path "/home/nlabiosa/llama13B/llama/vicuna-13b-v1.1" --data_path "/home/nlabiosa/LLaVA/ScienceQA/data/scienceqa/llava_minitest_QCM-LEPA.json" --output_path "/home/nlabiosa/llama13B/llama/generated_answers/ScienceQA_vicuna13B.jsonl"

#python CommonsenseQA/common_generate_answers.py --model_path "/home/nlabiosa/llama13B/llama/llama-7b-hf" --data_path "/home/nlabiosa/llama13B/llama/CommonsenseQA/train_rand_split.jsonl" --output_path "/home/nlabiosa/llama13B/llama/CommonsenseQA/llama7B.jsonl"
#python CommonsenseQA/common_generate_answers.py --model_path "/home/nlabiosa/llama13B/llama/llama-13b-hf" --data_path "/home/nlabiosa/llama13B/llama/CommonsenseQA/train_rand_split.jsonl" --output_path "/home/nlabiosa/llama13B/llama/CommonsenseQA/llama13B.jsonl"
#python CommonsenseQA/common_generate_answers.py --model_path "/home/nlabiosa/llama13B/llama/vicuna-7b-v1.1" --data_path "/home/nlabiosa/llama13B/llama/CommonsenseQA/train_rand_split.jsonl" --output_path "/home/nlabiosa/llama13B/llama/CommonsenseQA/vicuna7B.jsonl"
#python CommonsenseQA/common_generate_answers.py --model_path "/home/nlabiosa/llama13B/llama/vicuna-13b-v1.1" --data_path "/home/nlabiosa/llama13B/llama/CommonsenseQA/train_rand_split.jsonl" --output_path "/home/nlabiosa/llama13B/llama/CommonsenseQA/vicuna13B.jsonl"

sleep 3

echo "Ending time: $(date)" >> $benchmark_logs
echo "ulimit -l: " >> $benchmark_logs
ulimit -l >> $benchmark_logs

## Directory Cleanup ##
mv $job_name.$SLURM_JOB_ID.err $log_dir
mv $job_name.$SLURM_JOB_ID.out $log_dir

