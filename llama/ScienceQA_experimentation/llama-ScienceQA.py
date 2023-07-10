import json
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

# Open the JSON file
with open(args.data_path, 'r') as f:
    data = json.load(f)

# Open the output file
with open(args.output_path, 'a') as writer:
    # Iterate over each example in the dataset
    for i, example in enumerate(data):
        question = example['conversations'][0]
        qs = question['value']
        qs = qs.replace('<image>', '').strip()

        few_shot_prompt = ""

        for ex in examples:
            few_shot_prompt += f"Question: {ex['question']}\nThe answer is {ex['answer']}\n\n"

        # Concatenate the question and answer in a question-answer format
        prompt = few_shot_prompt + qs

        # Tokenize the prompt
        inputs = tokenizer.encode(prompt, return_tensors='pt')

        # Move the inputs to the device
        inputs = inputs.to(device)

        # Generate a response
        outputs = model.generate(inputs, max_length=750)

        # Decode the output tokens to text
        output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Remove the original prompt from the output
        output_text = output_text.replace(few_shot_prompt, '')

        # Add the generated response to the example
        example['model_answer'] = output_text

        # Write the example back to the JSONL file
        with open(args.output_path, 'a') as writer:
            writer.write(json.dumps(example) + '\n')

        # Print a status update
        if (i + 1) % 100 == 0:
            print(f"Processed {i + 1} examples")

print("Processing complete!")
