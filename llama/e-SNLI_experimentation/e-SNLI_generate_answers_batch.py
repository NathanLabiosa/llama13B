import json
import pandas as pd
import torch
from transformers import LlamaForCausalLM, LlamaTokenizer, TextGenerationPipeline
import argparse


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

# Create a TextGenerationPipeline
gen_pipeline = TextGenerationPipeline(
    model=model,
    tokenizer=tokenizer,
    device=0 if torch.cuda.is_available() else -1,
)

# Read the CSV file
df = pd.read_csv(args.data_path)

# List of hardcoded examples
examples = [
    {
        'gold_label': 'neutral',
        'Sentence1': 'A man in a green jersey and rollerskates stumbles as a man in a black jersey appears to collide with him.',
        'Sentence2': 'They both fall to the ground.',
        'Explanation_1': 'The combination of stumbling and colliding may not leave them both on the ground.'
    },
    {
        'gold_label': 'entailment',
        'Sentence1': 'Two young children in blue jerseys, one with the number 9 and one with the number 2 are standing on wooden steps in a bathroom and washing their hands in a sink.',
        'Sentence2': 'Two kids in numbered jerseys wash their hands.',
        'Explanation_1': 'Young children are kids. Jerseys with number 9 and 2 are numbered jerseys.'
    },
    {
        'gold_label': 'contradiction',
        'Sentence1': 'A young boy in a field of flowers carrying a ball',
        'Sentence2': 'dog in pool',
        'Explanation_1': 'The boy cannot also be a dog.'
    }

    # {
    #     'gold_label': 'contradiction',
    #     'Sentence1': 'Families waiting in line at an amusement park for their turn to ride.',
    #     'Sentence2': 'People are waiting to see a movie.',
    #     'Explanation_1': 'One does not usually see a movie at an amusement park.'
    # },

    # {
    #     'gold_label': 'entailment',
    #     'Sentence1': 'A man selling donuts to a customer during a world exhibition event held in the city of Angeles',
    #     'Sentence2': 'A man selling donuts to a customer.',
    #     'Explanation_1': 'A man selling donuts is selling donuts.'
    # },
    
    # {
    #     'gold_label': 'neutral',
    #     'Sentence1': 'Two women having drinks and smoking cigarettes at the bar.',
    #     'Sentence2': 'Women are celebrating at a bar.',
    #     'Explanation_1': 'Two women having drinks at a bar aren\'t necessarily celebrating.'
    # },
    # {
    #     'gold_label': 'entailment',
    #     'Sentence1': 'Two doctors perform surgery on patient.',
    #     'Sentence2': 'Doctors are performing surgery.',
    #     'Explanation_1': 'TWO  DOCTORS CAN ALSO BE DOCTORS.'
    # },
    
    # {
    #     'gold_label': 'entailment',
    #     'Sentence1': 'A group of onlookers glance at a person doing a strange trick on her head.',
    #     'Sentence2': 'People watch another person do a trick.',
    #     'Explanation_1': 'People watch a person doing a strange trick.'
    # },
    # {
    #     'gold_label': 'neutral',
    #     'Sentence1': 'Women and children sitting outside in the shade while a group of younger children sitting inside in the shade.',
    #     'Sentence2': 'It is very sunny outside.',
    #     'Explanation_1': 'Just because a group of people are sitting inside and outside the shade, does not mean that it is very sunny outside.'
    # },
    # {
    #     'gold_label': 'contradiction',
    #     'Sentence1': 'A man in a white t-shirt takes a picture in the middle of the street with two public buses in the background.',
    #     'Sentence2': 'A man is wearing a black shirt',
    #     'Explanation_1': 'White and black are two different colors.'
    # }

]




batch = []

for i, example in df.iterrows():
    Sentence1 = example['Sentence1']
    Sentence2 = example['Sentence2']
    label = example['gold_label']  # Convert the label to its corresponding relation

    # Combine the premise, hypothesis, and label in a question-answer format
    prompt = f"Given the two sentences: '{Sentence1}' '{Sentence2}', their relationship is "
    few_shot_prompt = ""

    # Prepend the prompt with a few examples
    for ex in examples:
        ex_Sentence1 = ex['Sentence1']
        ex_Sentence2 = ex['Sentence2']
        ex_label = ex['gold_label']

        few_shot_prompt = few_shot_prompt + (f"Given the two sentences: '{ex_Sentence1}' '{ex_Sentence2}', "
                    f"their relationship is '{ex_label}'. ")
    prompt = few_shot_prompt + prompt

    # Add our inputs to the batch
    batch.append((prompt, example))

    # If our batch is of the desired size or if this is the last example in the data
    if len(batch) == args.batch_size or i == len(df) - 1:
        # Prepare the batch prompts
        batch_prompts = [item[0] for item in batch]

        # Generate a response
        outputs = gen_pipeline(batch_prompts, max_length=400)

        for j, output in enumerate(outputs):
            # Decode the output tokens to text
            output_text = output[0]['generated_text']
            # Remove the original prompt from the output
            model_answer = output_text.replace(few_shot_prompt, '')
            print(model_answer, flush = True)

            # Convert the example to a dictionary and add the rationale
            example_dict = batch[j][1].to_dict()
            example_dict['model_answer'] = model_answer

            # Write the example (with the added rationale) to the JSON Lines file
            with open(args.output_path, 'a') as writer:
                writer.write(json.dumps(example_dict) + '\n')

        # Clear the batch for next round of processing
        batch = []