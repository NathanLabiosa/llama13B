import json
import re
from scipy.spatial.distance import cosine
from sentence_transformers import SentenceTransformer

# Load the sentence transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

correct_count = 0
incorrect_count = 0
invalid_count = 0
label_counts = {"entailment": 0, "neutral": 0, "contradiction": 0}
correct_label_counts = {"entailment": 0, "neutral": 0, "contradiction": 0}
invalid_rationales = []


def find_first_label_in_rationale(rationale, label_counts):
    words = re.findall(r"(entailment|neutral|contradiction)", rationale.lower())
    if words:
        return words[0]  # return the first label that appears
    return None  # if no labels appear, return None

with open("/home/nlabiosa/llama13B/llama/generated_answers/e-snli_vicuna13B_v2.jsonl", "r") as f:  # replace "your_file.jsonl" with your actual file name
    for i, line in enumerate(f):
        # if i >1000:
        #     break
        item = json.loads(line)
        rationale = item["model_answer"]
        label = item["gold_label"]
        if not rationale:
            invalid_count += 1
            label_counts[label] += 1
            continue
        first_label_in_rationale = find_first_label_in_rationale(rationale, label_counts)
        # Increase count for this label
        label_counts[label] += 1

        if first_label_in_rationale == label:
            correct_count += 1
            correct_label_counts[label] += 1  # Increase correct count for this label
        elif first_label_in_rationale in label_counts.keys():
            incorrect_count += 1
        else:
            invalid_count += 1
            invalid_rationales.append((rationale, label))

# Compute sentence embeddings for invalid rationales and labels
inval_rationales = [item[0] for item in invalid_rationales]
inval_embeddings = model.encode(inval_rationales)
label_embeddings = model.encode(list(label_counts.keys()))

inval_correct = 0
inval_incorrect = 0
# Compute cosine similarity for each invalid rationale
for i, (rationale, correct_label) in enumerate(invalid_rationales):
    similarities = []
    for label_embedding in label_embeddings:
        similarity = 1 - cosine(inval_embeddings[i], label_embedding)
        similarities.append(similarity)
    
    # Find the label with the highest similarity
    best_label_index = similarities.index(max(similarities))
    best_label = list(label_counts.keys())[best_label_index]

    # Check if the best label matches the correct label
    if best_label == correct_label:
        inval_correct += 1
    else:
        inval_incorrect += 1



print(f"Correct labels: {correct_count}", flush = True)
print(f"Incorrect labels: {incorrect_count}", flush = True)
print(f"Invalid labels: {invalid_count}", flush = True)
print(f"Invalid Correct labels: {inval_correct}", flush = True)
print(f"Invalid Incorrect labels: {inval_incorrect}", flush = True)

# for label, count in label_counts.items():
#     correct_count_for_label = correct_label_counts[label]
#     percentage_correct = (correct_count_for_label / count) * 100 if count > 0 else 0  # calculate percentage
#     print(f"Occurrences of label '{label}': {count}")
#     print(f"Correct predictions for label '{label}': {correct_count_for_label} ({percentage_correct:.2f}%)")
