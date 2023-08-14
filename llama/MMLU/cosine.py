import json
import os
from scipy.spatial.distance import cosine
from sentence_transformers import SentenceTransformer

# Load the sentence transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

categories = [
    "humanities",
    "STEM",
    "social sciences",
    "other (business, health, misc.)",
]

base_path = "/home/nlabiosa/llama13B/llama/MMLU/llava13B"

# Dictionary to store the counts for each category
category_counts = {category: {"correct": 0, "incorrect": 0} for category in categories}

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
                choices = item["choices"]

                # Compute sentence embedding for model answer
                model_answer_emb = model.encode([model_answer])[0]

                # Compute sentence embeddings for each choice and calculate cosine similarity
                similarities = []
                for choice in choices:
                    choice_emb = model.encode([choice["text"]])[0]
                    similarity = 1 - cosine(model_answer_emb, choice_emb)
                    similarities.append(similarity)

                # Find the choice with the highest similarity
                best_choice_index = similarities.index(max(similarities))
                best_choice = choices[best_choice_index]["label"]

                # Check if the best choice matches the correct answer
                if best_choice == true_answer:
                    category_counts[category]["correct"] += 1
                else:
                    category_counts[category]["incorrect"] += 1

total_accuracy = 0
num_categories = 0
# Print the results for each category
for category, counts in category_counts.items():
    total = counts['correct'] + counts['incorrect']
    if total > 0:
        accuracy = (counts['correct'] / total) * 100
        total_accuracy += accuracy
        num_categories += 1
    else:
        accuracy = 0
    print(f"{category}: correct {counts['correct']} incorrect {counts['incorrect']} accuracy {accuracy:.2f}%", flush=True)

if num_categories > 0:
    average_accuracy = total_accuracy / num_categories
else:
    average_accuracy = 0

print(f"Average accuracy across all categories: {average_accuracy:.2f}%", flush=True)
