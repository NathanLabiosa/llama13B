import json
import re

# Process the data
results = []
with open('/home/nlabiosa/LLaVA/finetuned/data/cose_minitrain_rationale_official_llama65B.jsonl', 'r') as f:
    i = 0
    for line in f:
        item = json.loads(line)

        # Parse the 'question' field if it's a stringified JSON
        if isinstance(item['question'], str):
            item['question'] = json.loads(item['question'])

        question = item['question']['stem']
        options = ', '.join([f'({choice["label"]}) {choice["text"]}' for choice in item['question']['choices']])
        model_answer = item['model_answer']

        prompt = f"{question}, Choices: {options},"

        # Split the model answer on the comma after (C)
        #print(prompt)
        split_prompt = model_answer[len(prompt)+3:]
       # print(split_prompt)

        # Split the model answer on the phrase "The correct answer is"
        split_answer = re.split(r'(The correct answer is [A-C] \w+)', split_prompt)
        #print(split_answer)


        # If the split was successful, take the second part
        if len(split_answer) > 1:
            model_answer = split_answer[0] + split_answer[1]

        output_data = {
            "id": item['id'],
            "conversations": [
                {
                    "from": "human",
                    "value": f'{question}\n{options}'
                },
                {
                    "from": "gpt",
                    "value": model_answer
                }
            ]
        }
        results.append(output_data)

# Save the results to a new file
with open('cos-e_65B_converted.json', 'w') as f:
    json.dump(results, f, indent=2)
