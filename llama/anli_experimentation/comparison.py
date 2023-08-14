import json
import re
from scipy.spatial.distance import cosine
from sentence_transformers import SentenceTransformer

# Load the sentence transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

correct_count = 0
incorrect_count = 0
invalid_count = 0
label_counts = {"e": 0, "n": 0, "c": 0}
correct_label_counts = {"e": 0, "n": 0, "c": 0}
label_mapping = {"e": "entailment", "n": "neutral", "c": "contradiction"}
inval = []

with open("/home/nlabiosa/llama13B/llama/generated_answers/anli_llava13B_v2.jsonl", "r") as f:
    for line in f:
        item = json.loads(line)
        rationale = item["rationale"]
        label = item["label"]
        if not rationale:
            invalid_count += 1
            label_counts[label] += 1
            continue

        first_word_rationale = None
        for word in ["entailment", "neutral", "contradiction"]:
            match = re.search(r'\b{}\b'.format(word), rationale, re.IGNORECASE)
            if match:
                first_word_rationale = match.group()
                break

        label = item["label"]
        label_word = label_mapping[label]  # this maps the label to its corresponding word
        # Increase count for this label
        label_counts[label] += 1

        if first_word_rationale and first_word_rationale.lower() == label_word.lower():
            correct_count += 1
            correct_label_counts[label] += 1  # Increase correct count for this label
        elif first_word_rationale and first_word_rationale.lower() in label_mapping.values():
            incorrect_count += 1
        else:
            invalid_count += 1
            inval.append((rationale, label))

# Compute sentence embeddings for invalid rationales and labels
inval_rationales = [item[0] for item in inval]
inval_embeddings = model.encode(inval_rationales)
label_embeddings = model.encode(list(label_mapping.values()))

inval_correct = 0
inval_incorrect = 0
# Compute cosine similarity for each invalid rationale
for i, (rationale, correct_label) in enumerate(inval):
    similarities = []
    for label_embedding in label_embeddings:
        similarity = 1 - cosine(inval_embeddings[i], label_embedding)
        similarities.append(similarity)
    
    # Find the label with the highest similarity
    best_label_index = similarities.index(max(similarities))
    best_label = list(label_mapping.keys())[best_label_index]

    # Check if the best label matches the correct label
    if best_label == correct_label:
        inval_correct += 1
    else:
        inval_incorrect += 1


print(f"Correct rationales: {correct_count}", flush = True)
print(f"Incorrect rationales: {incorrect_count}", flush = True)
print(f"Invalid rationales: {invalid_count}", flush = True)
print(f"invalid Correct rationales: {inval_correct}", flush = True)
print(f"invalid Incorrect rationales: {inval_incorrect}", flush = True)

# for label, count in label_counts.items():
#     correct_count_for_label = correct_label_counts[label]
#     percentage_correct = (correct_count_for_label / count) * 100 if count > 0 else 0  # calculate percentage
#     print(f"Occurrences of label '{label_mapping[label]}': {count}")
#     print(f"Correct predictions for label '{label_mapping[label]}': {correct_count_for_label} ({percentage_correct:.2f}%)")