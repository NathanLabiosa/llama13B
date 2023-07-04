import json
import jsonlines
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
        'question': 'Where can you buy binoculars?',
        'choices': [{'label': 'A', 'text': 'sporting goods store'}, {'label': 'B', 'text': 'suitcase'}, {'label': 'C', 'text': 'backpack'}],
        'label': 'A'
    },
    {
        'question': 'One of the potential hazards of attending school is what?',
        'choices': [{'label': 'A', 'text': 'taking tests'}, {'label': 'B', 'text': 'get smart'}, {'label': 'C', 'text': 'colds and flu'}],
        'label': 'C'
    },
    {
        'question': 'If somebody buys something and gives it to me as a free gift, what is the cost status of the gift?',
        'choices': [{'label': 'A', 'text': 'imprisoned'}, {'label': 'B', 'text': 'paid for'}, {'label': 'C', 'text': 'expensive'}],
        'label': 'B'
    }
]

few_shot_prompt = "\n\n".join([
    f"Given the question: '{example['question']}', the choices: {', '.join(['(' + choice['label'] + ') ' + choice['text'] for choice in example['choices']])}, the correct answer is ({example['label']}) {next(choice['text'] for choice in example['choices'] if choice['label'] == example['label'])}."
    for example in examples
])




# Open the JSONL file
with jsonlines.open(args.data_path, mode='r') as reader, jsonlines.open(args.output_path, mode='a') as writer:
    # Iterate over each example in the dataset
    for i, example in enumerate(reader):
        stem = example['question']['stem']
        answerKey = example['answerKey']

        # Convert the answerKey to answer text
        choices = example['question']['choices']
        for choice in choices:
            if choice['label'] == answerKey:
                answer = choice['text']

        # Format the choices text
        choices_text = ', '.join([f'({choice["label"]}) {choice["text"]}' for choice in choices])

        # Combine the stem and answer in a question-answer format
        prompt = f"{few_shot_prompt} Given the question: '{stem}', the choices: {choices_text}, the correct answer is"

        # Tokenize the prompt
        inputs = tokenizer.encode(prompt, return_tensors='pt')
        inputs = inputs.to(device)

        # Generate a response
        outputs = model.generate(inputs, max_length=400)

        # Decode the output tokens to text
        rationale = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Remove the few shot prompt from the rationale
        rationale = rationale[len(few_shot_prompt):]

        # Add the rationale to the example
        example['model_answer'] = rationale

        # Write the example back to the JSONL file
        with open(args.output_path, 'a') as writer:
            writer.write(json.dumps(example) + '\n')

        # Print a status update
        if (i + 1) % 100 == 0:
            print(f"Processed {i + 1} examples")

print("Processing complete!")
