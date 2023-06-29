# llama13B

## All Files
- environment.yml
  - This contains my conda environment "llava" which has all the necessary packages installed
- anli_generate_rationales.py
  - Generates rationales on the anli dataset
- anli_rationales.slurm
  - Slurm file for running anli_generate_rationales
  - The constraint, log_dir, and modules loaded are most likely the places to cause issues.
- e-SNLI_generate_rationales.py
  - Generates rationales on the e-SNLI dataset
- e-SNLI_rationales.slurm
  - Slurm file for running e-SNLI_generate_rationales
  - The constraint, log_dir, and modules loaded are most likely the places to cause issues.
- cos-e_generate_rationales.py
  - Generates rationales on the cos-e dataset
- cos-e_rationales.slurm
  - Slurm file for running cos-e_generate_rationales
  - The constraint, log_dir, and modules loaded are most likely the places to cause issues.
- SVAMP_generate_rationales.py
  - Generates rationales on the SVAMP dataset
- SVAMP_rationales.slurm
  - Slurm file for running SVAMP_generate_rationales
  - The constraint, log_dir, and modules loaded are most likely the places to cause issues.
- test_generate.py
  - Essentially the same as any generate_rationales.py file, just a place for me to run other experiments.
- llama-13b-hf
  - This is the 13B llama model I am using.
- job_logs
  - This is where my previous jobs are located

## Getting anli dataset
- Activate virtual environment
- Navigate to "llama13B/llama"
- Command: "git lfs install"
- Command: "git clone https://github.com/facebookresearch/anli"
- Navigate to "anli/scripts"
- Command: "bash download_data.sh"
- This will put the data in "anli/data/". The dataset should be good to go. (Note: we are using at "anli/data/anli_v1.0/R1/train.jsonl" for this rationale generation)

## Getting anli dataset
- Activate virtual environment
- Navigate to "llama13B/llama"
- Command: "git lfs install"
- Command: "git clone https://github.com/facebookresearch/anli"
- Navigate to "anli/scripts"
- Command: "bash download_data.sh"
- This will put the data in "anli/data/". The dataset should be good to go. (Note: we are using "anli/data/anli_v1.0/R1/train.jsonl" for this rationale generation)

## Getting e-SNLI dataset
- Activate virtual environment
- Navigate to "llama13B/llama"
- Command: "git lfs install"
- Command: "git clone https://github.com/OanaMariaCamburu/e-SNLI"
- Navigate to "e-SNLI/dataset/eSNLI"
- Command: "cat esnli_train_1.csv > esnli_train.csv"
- Command: "tail -n +2 esnli_train_2.csv >> esnli_train.csv"
- This will put all the training data into "esnli_train.csv" The dataset should be good to go.

## Getting cos-e dataset
- Activate virtual environment
- Navigate to "llama13B/llama"
- Command: "git lfs install"
- Command: "git clone https://github.com/salesforce/cos-e"
- The data we are working with is in "cos-e/data/v1.0/train_rand_split.jsonl"

## Getting SVAMP dataset
- Activate virtual environment
- Navigate to "llama13B/llama"
- Command: "git lfs install"
- Command: "git clone https://github.com/arkilpatel/SVAMP.git"
- The data we are working with is in "SVAMP/data/cv_asdiv-a/fold0/train.csv"

## Running
- Environment
  - set up the conda environment using the environment.yml file
### Anli
- anli_generate_rationales.py
  - change model_name to reflect your file path, the data path should be alright.
  - adjust "prompt" variable accordingly
- anli_rationales.slurm
  - change constraint, log_dir, modules loaded, any other changes to run slurm files on your system
  - Once you sbatch this slurm file, it will output in the "anli_rationales.jsonl"
### e_SNLI
- e-SNLI_generate_rationales.py
  - change model_name to reflect your file path, the data path should be alright
  - adjust "prompt" variable accordingly
- e-SNLI_rationales.slurm
  - change constraint, log_dir, modules loaded, any other changes to run slurm files on your system
  - Once you sbatch this slurm file, it will output in the "e-snli_rationales.csv"
### cos-e
- cos-e_generate_rationales.py
  - change model_name to reflect your file path, the data path should be alright
  - adjust "prompt" variable accordingly
- cos-e_rationales.slurm
  - change constraint, log_dir, modules loaded, any other changes to run slurm files on your system
  - Once you sbatch this slurm file, it will output in the "cos-e_rationales.jsonl"
### SVAMP
- SVAMP_generate_rationales.py
  - change model_name to reflect your file path, the data path should be alright
  - adjust "prompt" variable accordingly
  - I arbitrarily chose "fold0" as the data directory, this can be changed.
- SVAMP_rationales.slurm
  - change constraint, log_dir, modules loaded, any other changes to run slurm files on your system
  - Once you sbatch this slurm file, it will output in the "SVAMP_rationales.csv"
