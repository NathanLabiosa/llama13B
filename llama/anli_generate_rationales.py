import json
import jsonlines
import torch
from transformers import LlamaForCausalLM, LlamaTokenizer


# Load the LLMa model and tokenizer
model_name = "llama-13b-hf" # replace with the path to your model
tokenizer = LlamaTokenizer.from_pretrained(model_name)
model = LlamaForCausalLM.from_pretrained(model_name)

# Label mapping
label_mapping = {
    'e': "entailment",
    'n': "neutral",
    'c': "contradiction"
}

# Hardcoded examples for few-shot learning
examples = [
    {
        'context': 'Linguistics is the scientific study of language, and involves an analysis of language form, language meaning, and language in context. The earliest activities in the documentation and description of language have been attributed to the 4th century BCE Indian grammarian Pāṇini, who wrote a formal description of the Sanskrit language in his "Aṣṭādhyāyī".',
        'hypothesis': 'Form and meaning are the only aspects of language linguistics is concerned with.',
        'label': 'c',
        'rationale': 'Linguistics involves an analysis of language form, language meaning, and language in context, so context is also a crucial aspect.'
    },
    {
        'context': 'Franco Zeffirelli, KBE Grande Ufficiale OMRI (] ; born 12 February 1923) is an Italian director and producer of operas, films and television. He is also a former senator (1994–2001) for the Italian centre-right "Forza Italia" party. Recently, Italian researchers have found that he is one of the few distant relatives of Leonardo da Vinci.',
        'hypothesis': 'Franco Zeffirelli had a political career',
        'label': 'e',
        'rationale': 'Franco Zeffirelli was a senator so he had a political career.'
    },
]

# Format the few-shot examples as a prompt
few_shot_prompt = "\n\n".join([
    f"Given the context: '{example['context']}', the hypothesis: '{example['hypothesis']}', the answer is: '{label_mapping[example['label']]}' because {example['rationale']}"
    for example in examples
])


# Open the JSONL file
with jsonlines.open('anli/data/anli_v1.0/R1/train.jsonl', mode='r') as reader, open('anli_rationales.jsonl', 'a') as writer:
    # Iterate over each example in the dataset
    for i, example in enumerate(reader):
        context = example['context']
        hypothesis = example['hypothesis']
        answer = label_mapping[example['label']]  # Convert the label to its corresponding relation


        # Combine the context, hypothesis, and answer in a question-answer format
        #prompt = f"{few_shot_prompt}\n\nGiven the context: '{context}', the hypothesis: '{hypothesis}', answer in a sentence why the answer is: '{answer}'."
        prompt = f"{few_shot_prompt}\n\nGiven the context: '{context}', the hypothesis: '{hypothesis}', the answer is: '{answer}' because "
        
        #prompt = f"Given the context: '{context}', is the hypothesis: '{hypothesis}' correct? Why or why not?"

        # Tokenize the prompt
        inputs = tokenizer.encode(prompt, return_tensors='pt')

        # Generate a response
        outputs = model.generate(inputs, max_length=500)

        
        # Decode the output tokens to text
        rationale = tokenizer.decode(outputs[0], skip_special_tokens=True)

        rationale = rationale.replace(few_shot_prompt, '')

        # Add the rationale to the example
        example['rationale'] = rationale

        with open('anli_rationales.jsonl', 'a') as writer:
            writer.write(json.dumps(example) + '\n')
        # Print a status update
        if (i + 1) % 100 == 0:
            print(f"Processed {i + 1} examples")

print("Processing complete!")
