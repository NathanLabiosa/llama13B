import json
import re

correct_count = 0
incorrect_count = 0
invalid_count = 0
label_counts = {"e": 0, "n": 0, "c": 0}
correct_label_counts = {"e": 0, "n": 0, "c": 0}
label_mapping = {"e": "entailment", "n": "neutral", "c": "contradiction"}

with open("llava_anli_generate_fulltest.jsonl", "r") as f:
    for line in f:
        item = json.loads(line)
        rationale = item["rationale"]
        label = item["label"]
        if not rationale:
            invalid_count += 1
            label_counts[label] += 1
            continue
        first_word_rationale = rationale.split()[0]  # this splits the rationale into words and takes the first one
        first_word_rationale = re.sub(r'\W+', '', first_word_rationale)
        label = item["label"]
        label_word = label_mapping[label]  # this maps the label to its corresponding word
        # Increase count for this label
        label_counts[label] += 1

        if first_word_rationale.lower() == label_word.lower():
            correct_count += 1
            correct_label_counts[label] += 1  # Increase correct count for this label
        elif first_word_rationale.lower() in label_mapping.values():
            incorrect_count += 1
        else:
            invalid_count += 1

print(f"Correct rationales: {correct_count}", flush = True)
print(f"Incorrect rationales: {incorrect_count}", flush = True)
print(f"Invalid rationales: {invalid_count}", flush = True)

for label, count in label_counts.items():
    correct_count_for_label = correct_label_counts[label]
    percentage_correct = (correct_count_for_label / count) * 100 if count > 0 else 0  # calculate percentage
    print(f"Occurrences of label '{label_mapping[label]}': {count}")
    print(f"Correct predictions for label '{label_mapping[label]}': {correct_count_for_label} ({percentage_correct:.2f}%)")