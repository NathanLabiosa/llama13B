import json
from scipy.spatial.distance import cosine
from sentence_transformers import SentenceTransformer

# Load the sentence transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

correct_count = 0
incorrect_count = 0

with open("/home/nlabiosa/llama13B/llama/CommonsenseQA/blip_common.jsonl", "r") as f:
    for line in f:
        item = json.loads(line)
        model_answer = item["rationale"]
        true_answer = item["answerKey"]
        choices = item["question"]["choices"]

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
            correct_count += 1
        else:
            incorrect_count += 1

print(f"Correct answers: {correct_count}", flush = True)
print(f"Incorrect answers: {incorrect_count}", flush = True)
