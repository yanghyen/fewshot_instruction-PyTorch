# Few-Shot Prompting and Instruction Tuning Reproduction

This repository implements a compact reproduction framework for:

1. Brown et al. 2020, *Language Models are Few-Shot Learners*
2. Wei et al. 2022, *Finetuned Language Models Are Zero-Shot Learners*

The goal is not to rebuild GPT-3 or FLAN from scratch. Instead, this repo isolates the
main experimental ideas using open 7B-13B causal language models:

- in-context learning with `k={0,1,5}` demonstrations
- prompt-order randomization
- contextual calibration via label-probability normalization
- supervised instruction tuning with LoRA/QLoRA
- base model vs instruction-tuned model comparison
- instruction-present vs instruction-removed ablation
- reproducible metrics, runtime, VRAM, checkpoint, and failure-case logging

## Layout

```text
llm-adaptation-study/

├── src/
│
│   ├── data/
│   │   ├── piqa.py
│   │   ├── boolq.py
│   │   ├── hellaswag.py
│   │   └── arc.py
│
│   ├── models/
│   │   ├── qwen.py
│   │   ├── llama.py
│   │   └── base.py
│
│   ├── prompts/
│   │   ├── fewshot.py
│   │   ├── zeroshot.py
│   │   └── instruction.py
│
│   ├── evaluation/
│   │   ├── metrics.py
│   │   ├── calibration.py
│   │   └── runner.py
│
│   ├── fewshot/
│   │   └── evaluate.py
│
│   ├── instruction_tuning/
│   │   ├── train.py
│   │   ├── dataset_builder.py
│   │   └── evaluate.py
│
│   └── utils/
│       ├── seed.py
│       ├── logger.py
│       └── config.py
│
├── configs/
│
│   ├── fewshot/
│   │   ├── piqa/
│   │   ├── boolq/
│   │   └── hellaswag/
│
│   └── instruction_tuning/
│       ├── piqa/
│       └── multitask/
│
├── scripts/
│   ├── run_fewshot.sh
│   └── run_instruction_tuning.sh
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── cache/
│
├── outputs/
│   ├── predictions/
│   ├── metrics/
│   └── checkpoints/
│
├── tests/
│
└── README.md
```
