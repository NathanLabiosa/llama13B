import csv
import json

# with open('/home/nlabiosa/LLaVA/finetuned/data/svamp_rationale_official_llama65B.jsonl', 'r') as infile:
#     results = []
#     counter = 0
#     for line in infile:
#         #if counter >= 400:
#         #    break
#         data = json.loads(line)
#         numbers = data['Numbers'].split()
#         question = data['Question']

#         # replace each placeholder with the corresponding number
#         for i in range(min(len(numbers), 7)):
#             question = question.replace('number'+str(i), str(float(numbers[i])))

#         # Convert to the desired format
#         output_data = {
#             "id": str(counter),
#             "conversations": [
#                 {
#                     "from": "human",
#                     "value": question
#                 },
#                 {
#                     "from": "gpt",
#                     "value": "The answer is " + str(data['Answer'])  # adjusted this line
#                 }
#             ]
#         }

#         results.append(output_data)
#         counter += 1

# with open('/home/nlabiosa/LLaVA/finetuned/data/65B_converted.json', 'w') as outfile:
#     json.dump(results, outfile, indent=2) 


output_data = []
counter = 0

with open('/home/nlabiosa/llama13B/llama/SVAMP/data/cv_asdiv-a/fold0/train.csv', 'r') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        numbers = row['Numbers'].split()
        question = row['Question']

        # replace each placeholder with the corresponding number
        for i in range(min(len(numbers), 7)):
            question = question.replace('number'+str(i), str(float(numbers[i])))

        answer = row['Answer']
        output_data.append({
            "id": str(counter),
            "conversations": [
                {
                    "from": "human",
                    "value": question
                },
                {
                    "from": "gpt",
                    "value": "The answer is " + str(answer)
                }
            ]
        })
        counter += 1

# Write the output_data to a JSON file
with open('train_converted.json', 'w') as jsonfile:
    json.dump(output_data, jsonfile, indent=2)