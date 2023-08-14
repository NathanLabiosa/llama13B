import json
import re
from scipy.spatial.distance import cosine
from sentence_transformers import SentenceTransformer

# Load the sentence transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

correct_count = 0
incorrect_count = 0
label_counts = {"e": 0, "n": 0, "c": 0}
label_mapping = {"e": "entailment", "n": "neutral", "c": "contradiction"}

with open("/home/nlabiosa/llama13B/llama/generated_answers/anli_llava13B_v2.jsonl", "r") as f:
    # Compute sentence embeddings for labels
    label_embeddings = model.encode(list(label_mapping.values()))

    for line in f:
        item = json.loads(line)
        rationale = item["rationale"]
        label = item["label"]
        label_word = label_mapping[label]  # this maps the label to its corresponding word
        # Increase count for this label
        label_counts[label] += 1

        # Compute sentence embedding for rationale
        rationale_embedding = model.encode([rationale])[0]

        # Compute cosine similarity for the rationale
        similarities = []
        for label_embedding in label_embeddings:
            similarity = 1 - cosine(rationale_embedding, label_embedding)
            similarities.append(similarity)

        # Find the label with the highest similarity
        best_label_index = similarities.index(max(similarities))
        best_label = list(label_mapping.keys())[best_label_index]

        # Check if the best label matches the correct label
        if best_label == label:
            correct_count += 1
        else:
            incorrect_count += 1

print(f"Correct rationales: {correct_count}", flush = True)
print(f"Incorrect rationales: {incorrect_count}", flush = True)
