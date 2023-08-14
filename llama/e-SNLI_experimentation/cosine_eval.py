import json
import re
from scipy.spatial.distance import cosine
from sentence_transformers import SentenceTransformer

# Load the sentence transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

correct_count = 0
incorrect_count = 0
label_counts = {"entailment": 0, "neutral": 0, "contradiction": 0}

with open("/home/nlabiosa/llama13B/llama/generated_answers/e-snli_vicuna13B_v2.jsonl", "r") as f:
    # Compute sentence embeddings for labels
    label_embeddings = model.encode(list(label_counts.keys()))

    for i, line in enumerate(f):
        if i > 1000:
            break
        item = json.loads(line)
        rationale = item["model_answer"]
        label = item["gold_label"]

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
        best_label = list(label_counts.keys())[best_label_index]

        # Check if the best label matches the correct label
        if best_label == label:
            correct_count += 1
        else:
            incorrect_count += 1

print(f"Correct labels: {correct_count}", flush = True)
print(f"Incorrect labels: {incorrect_count}", flush = True)
