with open("train_examples.json", "r", encoding="utf-8") as f:
    train_examples = json.load(f)

filtered_examples = [
    ex for ex in train_examples
    if (
        "words" in ex and isinstance(ex["words"], list) and ex["words"] and
        "boxes" in ex and isinstance(ex["boxes"], list) and ex["boxes"] and
        "labels" in ex and isinstance(ex["labels"], list) and ex["labels"]
    )
]
print(f"Kept {len(filtered_examples)} good examples, filtered {len(train_examples) - len(filtered_examples)} bad ones.")
train_dataset = datasets.Dataset.from_list(filtered_examples)
