import json
import re

#Load the data from the JSON file
# with open('/home/nlabiosa/LLaVA/vqa_answers/minitest/specialized_output.jsonl') as f:
#     data = json.load(f)

with open('/home/nlabiosa/llama13B/llama/science_output.json') as f:
     data = json.load(f)
# # with open('rationale_testingoutput.json') as f:
# #      data = json.load(f)

# Initialize counters
total_correct_predictions = 0
total_incorrect_predictions = 0
total_invalid_predictions = 0
invalid = []
# Process "correct" and "incorrect" sections
for key in ["correct", "incorrect"]:
    for item in data[key]:
        ground_truth = item['ground_truth']
        pred_text = item['pred']

        # Extract the predicted answer from the 'pred' text, works best for LLaMA
        # idx = pred_text.find('The answer is')
        # if idx != -1:
        #     sub_string = pred_text[idx + len('The answer is'):].strip() # Removes whitespaces and newline characters
        #     if sub_string: # Make sure the string is not empty
        #         predicted_answer = sub_string.split()[0] # Split at first whitespace and take the first element
        #         predicted_answer = predicted_answer.rstrip('.')
        #         if ground_truth == predicted_answer:
        #             total_correct_predictions += 1
        #         else:
        #             total_incorrect_predictions += 1
        #     else:
        #         total_invalid_predictions += 1
        #         #invalid.append(item['pred'])
        # else:
        #     total_invalid_predictions += 1
            #invalid.append(item['pred'])

        # Works best for LLaVA
        #pattern = re.compile(r'\(([A-E])\)')
        # pattern = re.compile(r'The answer is \(?([A-E])\)?')
        pattern = re.compile(r'The answer is ([A-E])\.')
        match = pattern.search(pred_text)
        if match is not None:
            predicted_answer = match.group(1)

            if ground_truth == predicted_answer:
                total_correct_predictions += 1
            else:
                total_incorrect_predictions += 1
        else:
            total_invalid_predictions += 1

print(f"Total correct predictions: {total_correct_predictions}")
print(f"Total incorrect predictions: {total_incorrect_predictions}")
print(f"Total invalid predictions: {total_invalid_predictions}")
#print(invalid[:3])