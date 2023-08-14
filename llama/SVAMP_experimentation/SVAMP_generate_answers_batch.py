import json
import pandas as pd
import torch
from transformers import LlamaForCausalLM, LlamaTokenizer, TextGenerationPipeline
import argparse


# Command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str, required=True, help='Path to the model')
parser.add_argument('--data_path', type=str, required=True, help='Path to the CSV data')
parser.add_argument('--output_path', type=str, required=True, help='Path to the output file')
parser.add_argument('--batch_size', type=int, default=3, help='Batch size (default: 3)')
args = parser.parse_args()

# Load the LLMa model and tokenizer
tokenizer = LlamaTokenizer.from_pretrained(args.model_path)
model = LlamaForCausalLM.from_pretrained(args.model_path, torch_dtype=torch.float16, use_cache=True).cuda()


# Check if a GPU is available and if not, use a CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Move the model to the device
model = model.to(device)

# Create a TextGenerationPipeline
gen_pipeline = TextGenerationPipeline(
    model=model,
    tokenizer=tokenizer,
    device=0 if torch.cuda.is_available() else -1,
)

# Read the CSV file
df = pd.read_csv(args.data_path)  # replace with the path to your CSV file


examples = [

    {
        'question': '7 red apples and 2 green apples are in the basket . how many apples are in the basket ?',
        'answer': '9 apples .'
    },

    {
        'question': 'jake has 6 fewer peaches than steven . steven has 13 peaches . how many peaches does jake have ?',
        'answer': '7 peaches .'
    },

    {
        'question': 'kim has 4 cousins . she wants to give each one 5 pieces of gum . how much gum will she need ?',
        'answer': '20 pieces .'
    },
    {
        'question': 'mrs. hilt has 50 cents . a pencil costs 5 cents . how many pencils can she buy with the money she has ?',
        'answer': '10 pencils .'
    }
]

batch = []


# Iterate over each example in the dataset
for i, example in df.iterrows():
    # Split the numbers and create a mapping from 'numberN' to the corresponding number
    numbers = example['Numbers'].split()
    number_mapping = {f"number{n}": numbers[n] for n in range(len(numbers))}

    # Replace 'numberN' in the Question with the corresponding numbers
    question = example['Question']
    for number, value in number_mapping.items():
        question = question.replace(number, value)

    # Get the answer from the dataframe
    answer = example['Answer']

    few_shot_prompt = ""
    for ex in examples:
        ex_question = ex['question']
        ex_answer = ex['answer']
        prompt = f"\nGiven the question: '{ex_question}', the answer is: '{ex_answer}'."
        few_shot_prompt += prompt

    # Combine the question and answer into a prompt for the model
    prompt = few_shot_prompt + f"\nGiven the question: '{question}', the answer is: "

    # Add our inputs to the batch
    batch.append((prompt, example))

    # If our batch is of the desired size or if this is the last example in the data
    if len(batch) == args.batch_size or i == len(df) - 1:
        # Prepare the batch prompts
        batch_prompts = [item[0] for item in batch]

        # Generate a response
        outputs = gen_pipeline(batch_prompts, max_length=300)

        for j, output in enumerate(outputs):
            # Decode the output tokens to text
            output_text = output[0]['generated_text']
            # Remove the original prompt from the output
            model_answer = output_text.replace(few_shot_prompt, '')

            # Convert the example to a dictionary and add the rationale
            example_dict = batch[j][1].to_dict()
            example_dict['model_answer'] = model_answer

            # Write the example (with the added rationale) to the JSON Lines file
            with open(args.output_path, 'a') as writer:
                writer.write(json.dumps(example_dict) + '\n')

        # Clear the batch for next round of processing
        batch = []

