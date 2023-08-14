import json
import re

correct_count = 0
incorrect_count = 0
invalid_count = 0

with open("/home/nlabiosa/llama13B/llama/generated_answers/cos-e_vicuna13B.jsonl", "r") as f:
    for line in f:
        item = json.loads(line)
        model_answer = item["model_answer"]
        true_answer = item["answerKey"]
        # Look for the pattern "the answer is: .." in the model_answer string
        match = re.search(r"correct answer is \((.)\)", model_answer)
        if match:
            predicted_answer = match.group(1).strip()  # Extract the first matching group and remove any white spaces
            if predicted_answer == true_answer:
                correct_count += 1
            else:
                incorrect_count += 1
        else:
            invalid_count += 1

print(f"Correct answers: {correct_count}", flush = True)
print(f"Incorrect answers: {incorrect_count}", flush = True)
print(f"Invalid results: {invalid_count}", flush = True)

