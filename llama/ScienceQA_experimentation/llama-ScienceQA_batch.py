import json
import torch
from transformers import LlamaForCausalLM, LlamaTokenizer, TextGenerationPipeline
import argparse
import re

# Command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str, required=True, help='Path to the model')
parser.add_argument('--data_path', type=str, required=True, help='Path to the jsonl data')
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

examples = [
    {
        'question': "What information supports the conclusion that Olivia inherited this trait?\nContext: Read the description of a trait.\nOlivia has straight hair.\nOptions: (A) Olivia's neighbor also has straight hair. (B) Olivia's biological parents have red hair. Olivia also has red hair. (C) Olivia's biological mother often wears her straight hair in a ponytail.",
        'answer': "C"
    },
    {
        'question': "Based on the arrows, which of the following living things is a consumer?\nContext: Below is a food web from an ocean ecosystem. The ecosystem is in Monterey Bay, off the coast of California.\nA food web is a model that shows how the matter eaten by living things moves through an ecosystem. The arrows show how matter moves through the food web.\nOptions: (A) kelp (B) kelp bass",
        'answer': "B"
    },
    {
        'question': "What does the simile in this text suggest?\nTara rubbed coconut oil on her hands, which were like the parched earth during a drought.\nContext: N/A\nOptions: (A) Tara was baking something. (B) Tara's hands were dry and cracked.",
        'answer': "B"
    }
]

# Create a TextGenerationPipeline
gen_pipeline = TextGenerationPipeline(
    model=model,
    tokenizer=tokenizer,
    device=0 if torch.cuda.is_available() else -1,
)

# Open the JSON file
with open(args.data_path, 'r') as f:
    data = json.load(f)

# We'll use a list to store our batch
batch = []

# Iterate over each example in the dataset
for i, example in enumerate(data):
    example_id = example['id']
    question = example['conversations'][0]
    qs = question['value']
    qs = qs.replace('<image>', '').strip()

    few_shot_prompt = ""

    for ex in examples:
        few_shot_prompt += f"Question: {ex['question']}\nThe answer is {ex['answer']}\n\n"

    # Concatenate the question and answer in a question-answer format
    prompt = few_shot_prompt + qs

    # Add our inputs to the batch
    batch.append((prompt, example, example['id']))

    # If our batch is of the desired size or if this is the last example in the data
    if len(batch) == args.batch_size or i == len(data) - 1:
        # Prepare the batch prompts
        batch_prompts = [item[0] for item in batch]
        
        # Generate a response
        generated_texts = gen_pipeline(batch_prompts, max_length=500)

        # Process each example in the batch
        for j, generated_text in enumerate(generated_texts):
            # Extract the generated text
            #print(generated_text, flush = True)
            output_text = generated_text[0]['generated_text']

            # Remove the original prompt from the output
            output_text = output_text.replace(few_shot_prompt, '')

            # Split the output_text into lines
            lines = output_text.split('\n')

            human_input = '\n'.join(lines[:3])

            match = re.search(r'The answer is (.*)', output_text)

            if match:
                first_answer = "The answer is " + match.group(1)  # This will contain the text after the first "The answer is"
            else:
                first_answer = ""

            # Prepare output dictionary
            output = {
                "id": batch[j][2],
                "conversations": [
                    {"from": "human", "value": human_input},
                    {"from": "gpt", "value": first_answer},
                ],
                "text": first_answer
            }
            
            # Add the generated response to the example
            batch[j][1]['model_answer'] = output_text

            print(batch[j][1]['model_answer'], flush=True)
            print('end of model_answer')
            print(output, flush = True)
            print('end of output')
            # Write the example back to the JSONL file
            with open(args.output_path, 'a') as writer:
                writer.write(json.dumps(output) + '\n')

        # Clear the batch
        batch = []



