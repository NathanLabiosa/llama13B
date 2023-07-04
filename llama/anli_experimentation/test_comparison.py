import json
import re

correct_count = 0
incorrect_count = 0
invalid_count = 0
label_counts = {"e": 0, "n": 0, "c": 0}
correct_label_counts = {"e": 0, "n": 0, "c": 0}
label_mapping = {"e": "entailment", "n": "neutral", "c": "contradiction"}

def find_first_label_in_rationale(rationale, label_mapping):
    words = re.findall(r'\w+', rationale.lower())  # extract words from the rationale
    for word in words:
        for label, label_word in label_mapping.items():
            if word == label_word.lower():
                return label  # return the first label that appears
    return None  # if no labels appear, return None

with open("llava7B_anli_generate_minitest.jsonl", "r") as f:
    for line in f:
        item = json.loads(line)
        rationale = item["rationale"]
        label = item["label"]
        if not rationale:
            invalid_count += 1
            label_counts[label] += 1
            continue
        first_label_in_rationale = find_first_label_in_rationale(rationale, label_mapping)
        label_word = label_mapping[label]  # this maps the label to its corresponding word
        # Increase count for this label
        label_counts[label] += 1

        if first_label_in_rationale == label:
            correct_count += 1
            correct_label_counts[label] += 1  # Increase correct count for this label
        elif first_label_in_rationale in label_mapping.keys():
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

