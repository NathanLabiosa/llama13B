import json
import pandas as pd
import torch
from transformers import LlamaForCausalLM, LlamaTokenizer
import argparse


# Command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str, required=True, help='Path to the model')
parser.add_argument('--data_path', type=str, required=True, help='Path to the CSV data')
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
df = pd.read_csv(args.data_path)  # replace with the path to your CSV file


examples = [

    {
        'question': '7 red apples and 2 green apples are in the basket . how many apples are in the basket ?',
        'answer': 'There are 7 red and 2 green apples, so this is an addition problem. 7 plus 2 =  9 apples . The answer is 9 apples.'
    },

    {
        'question': 'jake has 6 fewer peaches than steven . steven has 13 peaches . how many peaches does jake have ?',
        'answer': 'jake has 13 minus 6 peaches, therefore this is a subtraction problem. he 13 minus 6 = 7 peaches . The answer is 7 peaches.'
    },

    {
        'question': 'kim has 4 cousins . she wants to give each one 5 pieces of gum . how much gum will she need ?',
        'answer': 'If each cousin gets 5 pieces and there are 4 cousins, this is a multiplication problem. 4 times 5 = 20 pieces . The answer is 20 pieces.'
    },
    {
        'question': 'mrs. hilt has 50 cents . a pencil costs 5 cents . how many pencils can she buy with the money she has ?',
        'answer': 'Each pencil will cost 5 cents and she has 50 total cents. This is a division problem. 50 divided by 5 = 10 pencils . The answer is 10 pencils. '
    }
]

# Open the output file
with open(args.output_path, 'w') as file:

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

            prompt = f"\n'{ex_question}' Think step by step and justify your answer. '{ex_answer}'"
            few_shot_prompt += prompt


        # Combine the question and answer into a prompt for the model
        prompt = few_shot_prompt + f"'{question}' Think step by step and justify your answer.  "

        # Tokenize the prompt
        inputs = tokenizer.encode(prompt, return_tensors='pt')
        inputs = inputs.to(device)

        # Generate a response
        outputs = model.generate(inputs, max_new_tokens=100)

        # Decode the output tokens to text
        model_answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

        model_answer = model_answer[len(few_shot_prompt):]
        # Convert the example to a dictionary and add the rationale
        example_dict = example.to_dict()
        example_dict['model_answer'] = model_answer

        # Write the example (with the added rationale) to the JSON Lines file
        with open(args.output_path, 'a') as writer:
            writer.write(json.dumps(example_dict) + '\n')




