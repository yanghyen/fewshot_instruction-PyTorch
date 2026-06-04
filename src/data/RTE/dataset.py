from datasets import load_dataset

dataset = load_dataset("nyu-mll/glue", "rte")

train_data = dataset["train"]
val_data = dataset["validation"]

# print(train_data[0])