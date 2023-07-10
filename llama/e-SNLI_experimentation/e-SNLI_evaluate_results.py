import json
import re

correct_count = 0
incorrect_count = 0
invalid_count = 0
label_counts = {"entailment": 0, "neutral": 0, "contradiction": 0}
correct_label_counts = {"entailment": 0, "neutral": 0, "contradiction": 0}

def find_first_label_in_rationale(rationale, label_counts):
    words = re.findall(r"(entailment|neutral|contradiction)", rationale.lower())
    if words:
        return words[0]  # return the first label that appears
    return None  # if no labels appear, return None

with open("e-snli_llava13B.jsonl", "r") as f:  # replace "your_file.jsonl" with your actual file name
    for line in f:
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

print(f"Correct labels: {correct_count}", flush = True)
print(f"Incorrect labels: {incorrect_count}", flush = True)
print(f"Invalid labels: {invalid_count}", flush = True)

for label, count in label_counts.items():
    correct_count_for_label = correct_label_counts[label]
    percentage_correct = (correct_count_for_label / count) * 100 if count > 0 else 0  # calculate percentage
    print(f"Occurrences of label '{label}': {count}")
    print(f"Correct predictions for label '{label}': {correct_count_for_label} ({percentage_correct:.2f}%)")
