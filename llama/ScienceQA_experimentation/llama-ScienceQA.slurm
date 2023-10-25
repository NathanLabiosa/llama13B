#!/bin/bash
#
##SBATCH --nodes=1
#SBATCH -p gpu
#SBATCH --cpus-per-task=4
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:1
#SBATCH --job-name=llama_ScienceQA

#SBATCH --error=llama_ScienceQA.%J.err
#SBATCH --output=llama_ScienceQA.%J.out

export job_name="llama_ScienceQA"
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
# python llama-ScienceQA.py --model_path "/home/nlabiosa/llama13B/llama/llama-7b-hf" --data_path "/home/nlabiosa/LLaVA/ScienceQA/data/scienceqa/llava_test_QCM-LEPA.json" --output_path "/home/nlabiosa/llama13B/llama/ScienceQA_experimentation/fulltest4/new_llama7B.jsonl"
# python llama-ScienceQA.py --model_path "/home/nlabiosa/llama13B/llama/llama-13b-hf" --data_path "/home/nlabiosa/LLaVA/ScienceQA/data/scienceqa/llava_test_QCM-LEPA.json" --output_path "/home/nlabiosa/llama13B/llama/ScienceQA_experimentation/fulltest4/new_llama13B.jsonl"
#python blip_noimage.py
# python blip_image.py --model_path /home/nlabiosa/llama13B/llama/blip --data_path "/home/nlabiosa/LLaVA/ScienceQA/data/scienceqa/llava_test_QCM-LEPA.json" --output_path "/home/nlabiosa/llama13B/llama/ScienceQA_experimentation/fulltest4/blip_noimage.jsonl"

# python eval.py --data_path /home/nlabiosa/llama13B/llama/ScienceQA_experimentation/fulltest2/llava7B_image.jsonl --split test
# python eval.py --data_path /home/nlabiosa/llama13B/llama/ScienceQA_experimentation/fulltest2/llava7B_noimage.jsonl --split test
# python eval.py --data_path /home/nlabiosa/llama13B/llama/ScienceQA_experimentation/llava13B_image.jsonl --split filtered
# python eval.py --data_path /home/nlabiosa/llama13B/llama/ScienceQA_experimentation/llava13B_noimage.jsonl --split filtered
# python eval.py --data_path /home/nlabiosa/llama13B/llama/ScienceQA_experimentation/fulltest4/llava7B_image.jsonl --ground_truth_path /home/nlabiosa/llama13B/llama/ScienceQA_experimentation/groundtruth_full.jsonl
# python eval.py --data_path /home/nlabiosa/llama13B/llama/ScienceQA_experimentation/fulltest4/llava7B_noimage.jsonl  --ground_truth_path /home/nlabiosa/llama13B/llama/ScienceQA_experimentation/groundtruth_full.jsonl
# python eval.py --data_path /home/nlabiosa/llama13B/llama/ScienceQA_experimentation/fulltest4/llava13B_image.jsonl --ground_truth_path /home/nlabiosa/llama13B/llama/ScienceQA_experimentation/groundtruth_full.jsonl
# python eval.py --data_path /home/nlabiosa/llama13B/llama/ScienceQA_experimentation/fulltest4/llava13B_noimage.jsonl  --ground_truth_path /home/nlabiosa/llama13B/llama/ScienceQA_experimentation/groundtruth_full.jsonl

# python eval.py --data_path /home/nlabiosa/llama13B/llama/ScienceQA_experimentation/fulltest3/llava7B_image.jsonl --ground_truth_path /home/nlabiosa/llama13B/llama/ScienceQA_experimentation/groundtruth_full.jsonl
# python eval.py --data_path /home/nlabiosa/llama13B/llama/ScienceQA_experimentation/fulltest3/llava7B_noimage.jsonl  --ground_truth_path /home/nlabiosa/llama13B/llama/ScienceQA_experimentation/groundtruth_full.jsonl
# python eval.py --data_path /home/nlabiosa/llama13B/llama/ScienceQA_experimentation/fulltest3/llava13B_image.jsonl --ground_truth_path /home/nlabiosa/llama13B/llama/ScienceQA_experimentation/groundtruth_full.jsonl
# python eval.py --data_path /home/nlabiosa/llama13B/llama/ScienceQA_experimentation/fulltest3/llava13B_noimage.jsonl  --ground_truth_path /home/nlabiosa/llama13B/llama/ScienceQA_experimentation/groundtruth_full.jsonl

python cosine.py --data_path /home/nlabiosa/llama13B/llama/ScienceQA_experimentation/fulltest4/llama_llava.jsonl --ground_truth_path /home/nlabiosa/llama13B/llama/ScienceQA_experimentation/groundtruth_full.jsonl
#python cosine.py --data_path /home/nlabiosa/llama13B/llama/ScienceQA_experimentation/fulltest4/control2.jsonl --ground_truth_path /home/nlabiosa/llama13B/llama/ScienceQA_experimentation/groundtruth_full.jsonl
#python cosine.py --data_path /home/nlabiosa/llama13B/llama/ScienceQA_experimentation/fulltest4/onlytext_output --ground_truth_path /home/nlabiosa/llama13B/llama/ScienceQA_experimentation/groundtruth_full.jsonl
#python cosine.py --data_path /home/nlabiosa/llama13B/llama/ScienceQA_experimentation/fulltest4/onlyimages_output --ground_truth_path /home/nlabiosa/llama13B/llama/ScienceQA_experimentation/groundtruth_full.jsonl
# python eval.py --data_path /home/nlabiosa/llama13B/llama/ScienceQA_experimentation/fulltest2/llava7B_noimage.jsonl  --ground_truth_path /home/nlabiosa/llama13B/llama/ScienceQA_experimentation/groundtruth_full.jsonl
# python eval.py --data_path /home/nlabiosa/llama13B/llama/ScienceQA_experimentation/fulltest2/llava13B_image.jsonl --ground_truth_path /home/nlabiosa/llama13B/llama/ScienceQA_experimentation/groundtruth_full.jsonl
# python eval.py --data_path /home/nlabiosa/llama13B/llama/ScienceQA_experimentation/fulltest2/llava13B_noimage.jsonl  --ground_truth_path /home/nlabiosa/llama13B/llama/ScienceQA_experimentation/groundtruth_full.jsonl

# python VisualBERT.py --data_path "/home/nlabiosa/LLaVA/ScienceQA/data/scienceqa/llava_test_QCM-LEPA.json" --output_path "/home/nlabiosa/llama13B/llama/ScienceQA_experimentation/VisualBERT/VisualBERT_images.jsonl" --image_folder /home/nlabiosa/LLaVA/ScienceQA/data/scienceqa/test

sleep 3

echo "Ending time: $(date)" >> $benchmark_logs
echo "ulimit -l: " >> $benchmark_logs
ulimit -l >> $benchmark_logs

## Directory Cleanup ##
mv $job_name.$SLURM_JOB_ID.err $log_dir
mv $job_name.$SLURM_JOB_ID.out $log_dir

