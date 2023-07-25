import json
import pandas as pd
import torch
from transformers import LlamaForCausalLM, LlamaTokenizer
import argparse


# Command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str, required=True, help='Path to the model')
parser.add_argument('--data_path', type=str, required=True, help='Path to the jsonl data')
parser.add_argument('--output_path', type=str, required=True, help='Path to the output file')
args = parser.parse_args()

# Load the LLMa model and tokenizer
tokenizer = LlamaTokenizer.from_pretrained(args.model_path)
model = LlamaForCausalLM.from_pretrained(args.model_path, torch_dtype=torch.float16, use_cache=True).cuda()

# Check if a GPU is available and if not, use a CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Move the model to the device
model = model.to(device)

# Read the CSV file
df = pd.read_csv(args.data_path)

# List of hardcoded examples
examples = [
    {
        'gold_label': 'entailment',
        'Sentence1': 'Two young children in blue jerseys, one with the number 9 and one with the number 2 are standing on wooden steps in a bathroom and washing their hands in a sink.',
        'Sentence2': 'Two kids in numbered jerseys wash their hands.',
        'Explanation_1': 'Young children are kids. Jerseys with number 9 and 2 are numbered jerseys.'
    },
    {
        'gold_label': 'entailment',
        'Sentence1': 'A man selling donuts to a customer during a world exhibition event held in the city of Angeles',
        'Sentence2': 'A man selling donuts to a customer.',
        'Explanation_1': 'A man selling donuts is selling donuts.'
    },
    {
        'gold_label': 'contradiction',
        'Sentence1': 'A young boy in a field of flowers carrying a ball',
        'Sentence2': 'dog in pool',
        'Explanation_1': 'The boy cannot also be a dog.'
    }
]


# Open the output file
with open(args.output_path, 'w') as file:

    # Iterate over each example in the dataset
    for i, example in df.iterrows():
        Sentence1 = example['Sentence1']
        Sentence2 = example['Sentence2']
        label = example['gold_label']  # Convert the label to its corresponding relation

        # Combine the premise, hypothesis, and label in a question-answer format
        replace_prompt = f"Given the two sentences: '{Sentence1}', and '{Sentence2}', what is their relationship to each other? Think step by step and justify your steps. "
        few_shot_prompt = ""
        # Prepend the prompt with a few examples
        for ex in examples:
            ex_Sentence1 = ex['Sentence1']
            ex_Sentence2 = ex['Sentence2']
            ex_label = ex['gold_label']
            ex_explaination = ex['Explanation_1']

            few_shot_prompt = few_shot_prompt + (f"Given the two sentences: '{Sentence1}', and '{Sentence2}', what is their relationship to each other? Think step by step and justify your steps. '{ex_explaination}' Therefore their relationship is '{ex_label}' ")
        
        prompt = few_shot_prompt + replace_prompt
        # Tokenize the prompt
        inputs = tokenizer.encode(prompt, return_tensors='pt')
        inputs = inputs.to(device)
        
        # Generate a response
        outputs = model.generate(inputs, max_new_tokens = 50)

        # Decode the output tokens to text
        model_answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

        model_answer = model_answer.replace(few_shot_prompt, '')
        model_answer = model_answer.replace(replace_prompt, '')
        # Convert the example to a dictionary and add the rationale
        example_dict = example.to_dict()
        example_dict['model_answer'] = model_answer

        # Write the example (with the added rationale) to the JSON Lines file
        with open(args.output_path, 'a') as writer:
            writer.write(json.dumps(example_dict) + '\n')

