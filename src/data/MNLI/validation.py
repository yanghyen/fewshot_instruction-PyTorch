import random
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

SRC_ROOT = Path(__file__).resolve().parents[2]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from data.MNLI.dataset import dataset
from data.MNLI.prompt import CHOICE_LABELS, build_prompt, score_answer
from models.load_model import model, tokenizer

train_data = dataset["train"]
valid_data = dataset["validation_matched"]

shots = [0, 1, 5]
seeds = [42, 43, 44]
num_eval = len(valid_data)
# num_eval = 30

eval_examples = valid_data.select(range(num_eval))
train_examples = list(train_data)

results = []

for seed in seeds:
    random.seed(seed)

    for shot in shots:
        correct = 0
        total = 0

        for idx, example in enumerate(tqdm(eval_examples, desc=f"seed {seed} | {shot}-shot")):
            demo_indices = random.sample(range(len(train_examples)), shot)
            demo_examples = [train_examples[i] for i in demo_indices]

            prompt = build_prompt(
                test_example=example,
                demo_examples=demo_examples,
            )

            scores = [
                score_answer(prompt, f" {choice}", model, tokenizer)
                for choice in CHOICE_LABELS
            ]

            pred_label = max(range(len(scores)), key=lambda i: scores[i])
            gold_label = int(example["label"])

            is_correct = pred_label == gold_label

            if is_correct:
                correct += 1

            total += 1

            results.append({
                "seed": seed,
                "shot": shot,
                "example_id": idx,
                "premise": example["premise"],
                "hypothesis": example["hypothesis"],
                "gold_label": gold_label,
                "pred_label": pred_label,
                "score_a": scores[0],
                "score_b": scores[1],
                "score_c": scores[2],
                "demo_indices": demo_indices,
                "is_correct": is_correct,
                "total": total,
                "correct_cnt": correct,
            })

        final_accuracy = correct / total
        print(
            f"seed {seed} | {shot}-shot Final Acc: "
            f"{final_accuracy:.4f} ({correct}/{total})"
        )

        df = pd.DataFrame(results)
        df.to_csv("mnli_qwen_results.csv", index=False)
