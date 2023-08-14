import json
import re
import os

categories = [
    "humanities",
    "STEM",
    "social sciences",
    "other (business, health, misc.)",
]

base_path = "/home/nlabiosa/llama13B/llama/MMLU/llava13B"

# Dictionary to store the counts for each category
category_counts = {category: {"correct": 0, "incorrect": 0, "invalid": 0} for category in categories}

# Loop through the categories
for category in categories:
    category_path = os.path.join(base_path, category)
    
    # Loop through the result files within the category folder
    for result_file in os.listdir(category_path):
        result_file_path = os.path.join(category_path, result_file)
        
        with open(result_file_path, "r") as f:
            for line in f:
                item = json.loads(line)
                model_answer = item["model_answer"]
                true_answer = item["answerKey"]
                # Look for the pattern "the answer is: .." in the model_answer string
                #match = re.search(r"correct answer is \((.)\)", model_answer)
                match = re.search(r"\((.)\)", model_answer)
                if match:
                    predicted_answer = match.group(1).strip()  # Extract the first matching group and remove any white spaces
                    if predicted_answer == true_answer:
                        category_counts[category]["correct"] += 1
                    else:
                        category_counts[category]["incorrect"] += 1
                else:
                    category_counts[category]["invalid"] += 1


total_accuracy = 0
num_categories = 0
# Print the results for each category
for category, counts in category_counts.items():
    total = counts['correct'] + counts['incorrect'] + counts['invalid']
    if total > 0:
        accuracy = (counts['correct'] / total) * 100
        total_accuracy += accuracy
        num_categories += 1
    else:
        accuracy = 0
    print(f"{category}: correct {counts['correct']} incorrect {counts['incorrect']} invalid {counts['invalid']} accuracy {accuracy:.2f}%", flush=True)

if num_categories > 0:
    average_accuracy = total_accuracy / num_categories
else:
    average_accuracy = 0

print(f"Average accuracy across all categories: {average_accuracy:.2f}%", flush=True)
print("",flush=True)