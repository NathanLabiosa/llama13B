import json
import re

# Initialize counters
total_correct_predictions = 0
total_incorrect_predictions = 0
total_invalid_predictions = 0

# Specify the correct answers
correct_answer_pattern = r'^The correct caption is \(A\)[,.]?\s*(.*)'


# Read and process the file
with open('/home/nlabiosa/llama13B/llama/winoground_experimentation/llama7B_winoground_answers.jsonl', 'r') as file:
    for line in file:
        data = json.loads(line)

        text = data["text"]

        # Check if the model's output matches the expected answer
        if re.search(correct_answer_pattern, text):
            total_correct_predictions += 1
        elif re.search(r'The correct caption is \(B\)[,.]?\s*(.*)', text):
            total_incorrect_predictions += 1
        else:
            total_invalid_predictions += 1

print(f"Total correct predictions: {total_correct_predictions}")
print(f"Total incorrect predictions: {total_incorrect_predictions}")
print(f"Total invalid predictions: {total_invalid_predictions}")
