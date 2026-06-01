from datasets import DatasetDict, load_dataset

BASE_URL = "https://yonatanbisk.com/piqa/data"

examples = load_dataset(
    "json",
    data_files={
        "train": f"{BASE_URL}/train.jsonl",
        "validation": f"{BASE_URL}/valid.jsonl",
        "test": f"{BASE_URL}/tests.jsonl",
    },
)

labels = load_dataset(
    "text",
    data_files={
        "train": f"{BASE_URL}/train-labels.lst",
        "validation": f"{BASE_URL}/valid-labels.lst",
    },
)


def add_labels(split):
    split_labels = [int(row["text"]) for row in labels[split]]
    return examples[split].add_column("label", split_labels)


dataset = DatasetDict(
    {
        "train": add_labels("train"),
        "validation": add_labels("validation"),
        "test": examples["test"].add_column("label", [-1] * len(examples["test"])),
    }
)

# print(dataset)
# print(dataset["train"][0])
# print(dataset["validation"][0])