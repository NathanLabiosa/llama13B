# llama13B

## All Files
- environment.yml
  - This contains my conda environment "llava" which has all the necessary packages installed
- generate_rationales.py
  -This is the file where all the generation happens.
    - You will have to change the "model_name" parameter
    - The dataset path should be acceptable, but also something to be aware of
    - The "prompt" variable is where most of the experimentation occurs. I have it as one option right now, but swapping it for the above line should also produce rationales.
    - It puts all the outputs into the "rationales.jsonl" file
- test_generate.py
  - Essentially the same as generate_rationales.py, just a place for me. to run other experiments.
- rationales.slurm
  - I'm not sure how much your system differs in the slurm file.
  - The constraint, log_dir, and modules loaded are most likely the places to cause issues.
- llama-13b-hf
  - This is the 13B llama model I am using.
- job_logs
  - This is where my previous jobs are located

## Getting anli dataset
- Activate virtual environment
- Command: "git lfs install"
- Command: "git clone https://github.com/facebookresearch/anli"
- Navigate to "anli/scripts"
- Command: "bash download_data.sh"
- This will put the data in "anli/data/". We are looking at "anli/data/anli_v1.0/R1/train.jsonl" for this rationale generation

## Running
- Environment
  - set up the conda environment using the environment.yml file
- generate_rationales.py
  - change model_name to reflect your file path, the data path should be alright.
  - adjust "prompt" variable accordingly
- rationales.slurm
  - change constraint, log_dir, modules loaded, any other changes to run slurm files on your system
  - Once you sbatch this slurm file, it will output in the "rationales.jsonl"
