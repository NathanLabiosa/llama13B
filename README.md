# llama13B

## All Directories and Important Files
- __environment.yml__
  - This contains my conda environment "llava" which has all the necessary packages installed
- __anli__
  - The anli dataset
- __anli_experimentation__
  - All my code and results for running tests on the anli dataset
- __cos-e__
  - The cos-e dataset
- __cos-e_experimentation__
  - All my code and results for running tests on the cos-e dataset
- __e-SNLI__
  - The e-SNLI dataset
- __e-SNLI_experimentation__
  - All my code and results for running tests on the e-SNLI dataset
- __SVAMP__
  - the SVAMP dataset
- __SVAMP_experimentation__
  - All my code and results for running tests on the SVAMP dataset
- __generated_answers__
  - For holding more extensive experiment data temporarily before moving to their respective _experimentation folder
- __run_all_answer_generation.slurm__
  - Made for mass experiments. Runs all data through a specified model to get output
- __llama-13b-hf__
  - This is the 13B llama model I am using.
- __llama-7b-hf__
  - This is the 7B llama model I am using.
- __job_logs__
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
### Updated 07/04/2023
- Once the datasets and environment are set up, only changes to the "run_all_answer_generation.slurm" must be made.
  - These changes include your model path, data path, and output path. If you can change the output name to reflect the model you are using, aka llama65B, that would be great. Any other changes to the slurm file will be for compatibility on your network.
- After doing this, the job should run through all four datasets smoothly, putting all outputs in your output path. I would suggest keeping it as the generated_answers directory so it's all in once place.

