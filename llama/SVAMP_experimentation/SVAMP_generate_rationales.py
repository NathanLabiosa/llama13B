import json
import pandas as pd
from transformers import LlamaForCausalLM, LlamaTokenizer

# Load the LLMa model and tokenizer
model_name = "llama-13b-hf"  # replace with the path to your model
tokenizer = LlamaTokenizer.from_pretrained(model_name)
model = LlamaForCausalLM.from_pretrained(model_name)

# Read the CSV file
df = pd.read_csv('/home/nlabiosa/llama13B/llama/SVAMP/data/cv_asdiv-a/fold0/train.csv')  # replace with the path to your CSV file

# Open the output file
with open('SVAMP_rationales.jsonl', 'w') as file:

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

        # Combine the question and answer into a prompt for the model
        prompt = f"The question is: '{question}', and the answer is: '{answer}'. Can you explain why this is the answer?"

        # Tokenize the prompt
        inputs = tokenizer.encode(prompt, return_tensors='pt')

        # Generate a response
        outputs = model.generate(inputs, max_length=200)

        # Decode the output tokens to text
        rationale = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Convert the example to a dictionary and add the rationale
        example_dict = example.to_dict()
        example_dict['Rationale'] = rationale

        # Write the example (with the added rationale) to the JSON Lines file
        with open('SVAMP_rationales.jsonl', 'a') as writer:
            writer.write(json.dumps(example_dict) + '\n')

        # Print a status update
        if (i + 1) % 100 == 0:
            print(f"Processed {i + 1} examples")

print("Processing complete!")
