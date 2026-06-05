import random
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

SRC_ROOT = Path(__file__).resolve().parents[2]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from data.RTE.dataset import dataset
from data.RTE.prompt import build_prompt, score_answer
from models.load_model import model, tokenizer

train_data = dataset["train"]
valid_data = dataset["validation"]

shots = [1, 5]
demo_seeds = [42, 43, 44]
order_seeds = [0, 1, 2, 3, 4]
num_eval = len(valid_data)
# num_eval = 10

eval_examples = valid_data.select(range(num_eval))
train_examples = list(train_data)

results = []

for demo_seed in demo_seeds:
    demo_rng = random.Random(demo_seed)

    for shot in shots:
        base_demo_indices_by_example = [
            demo_rng.sample(range(len(train_examples)), shot)
            for _ in range(num_eval)
        ]

        for order_seed in order_seeds:
            correct = 0
            total = 0
            order_rng = random.Random(order_seed)

            desc = f"demo_seed {demo_seed} | order_seed {order_seed} | {shot}-shot"
            for idx, example in enumerate(tqdm(eval_examples, desc=desc)):
                demo_indices = base_demo_indices_by_example[idx]
                demo_order = demo_indices.copy()
                order_rng.shuffle(demo_order)
                demo_examples = [train_examples[i] for i in demo_order]

                prompt = build_prompt(
                    test_example=example,
                    demo_examples=demo_examples,
                )

                score_a = score_answer(prompt, " A", model, tokenizer)
                score_b = score_answer(prompt, " B", model, tokenizer)
                pred_label = 0 if score_a > score_b else 1
                gold_label = int(example["label"])

                is_correct = pred_label == gold_label

                if is_correct:
                    correct += 1

                total += 1

                results.append({
                    "demo_seed": demo_seed,
                    "order_seed": order_seed,
                    "shot": shot,
                    "example_id": idx,
                    "sentence1": example["sentence1"],
                    "sentence2": example["sentence2"],
                    "gold_label": gold_label,
                    "pred_label": pred_label,
                    "score_a": score_a,
                    "score_b": score_b,
                    "demo_indices": demo_indices,
                    "demo_order": demo_order,
                    "is_correct": is_correct,
                    "total": total,
                    "correct_cnt": correct,
                })

            final_accuracy = correct / total
            print(
                f"demo_seed {demo_seed} | order_seed {order_seed} | {shot}-shot "
                f"Final Acc: {final_accuracy:.4f} ({correct}/{total})"
            )

            df = pd.DataFrame(results)
            df.to_csv("rte_ablation1_qwen_results.csv", index=False)
