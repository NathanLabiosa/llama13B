import json
import pandas as pd
from transformers import LlamaForCausalLM, LlamaTokenizer

# Load the LLMa model and tokenizer
model_name = "llama-13b-hf" # replace with the path to your model
tokenizer = LlamaTokenizer.from_pretrained(model_name)
model = LlamaForCausalLM.from_pretrained(model_name)


# Read the CSV file
df = pd.read_csv('e-SNLI/dataset/esnli_train.csv')

# Open the output file
with open('e-snli_rationales.jsonl', 'w') as file:

    # Iterate over each example in the dataset
    for i, example in df.iterrows():
        Sentence1 = example['Sentence1']
        Sentence2 = example['Sentence2']
        label = example['gold_label']  # Convert the label to its corresponding relation

        # Combine the premise, hypothesis, and label in a question-answer format
        prompt = f"Given the two sentences: '{Sentence1}' '{Sentence2}', why can the relation between the two sentences be described as: '{label}'?"

        # Tokenize the prompt
        inputs = tokenizer.encode(prompt, return_tensors='pt')

        # Generate a response
        outputs = model.generate(inputs, max_length=200)

        # Decode the output tokens to text
        rationale = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Convert the example to a dictionary and add the rationale
        example_dict = example.to_dict()
        example_dict['rationale'] = rationale

        # Write the example (with the added rationale) to the JSON Lines file
        with open('e-snli_rationales.jsonl', 'a') as writer:
            writer.write(json.dumps(example_dict) + '\n')

        # Print a status update
        if (i + 1) % 100 == 0:
            print(f"Processed {i + 1} examples")

print("Processing complete!")
