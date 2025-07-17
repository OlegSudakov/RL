import argparse
import os

from datasets import load_dataset, concatenate_datasets

parser = argparse.ArgumentParser()
parser.add_argument("--test_size", type=float, required=False, default=0.01)
parser.add_argument("--seed", type=int, required=False, default=0)
parser.add_argument("--out_path", type=str, required=False, default=".")
parser.add_argument("--num_proc", type=int, required=False, default=32)


def format_llama_nemotron_sample(sample: dict[str, str]):
    messages = [{"role": "system", "content": sample["system_prompt"]}]
    messages += sample["input"]
    messages += [{"role": "assistant", "content": sample["output"]}]
    return {
        "messages": messages,
        "task_name": sample["category"]
    }


if __name__ == "__main__":
    args = parser.parse_args()
    original_ds = load_dataset("nvidia/Llama-Nemotron-Post-Training-Dataset", "SFT")
    original_ds = concatenate_datasets([dataset for dataset in original_ds.values()])
    split_ds = original_ds.train_test_split(test_size=args.test_size, seed=args.seed)

    train_formatted = split_ds["train"].map(
        format_llama_nemotron_sample,
        remove_columns=split_ds["train"].column_names,
        num_proc=min(os.cpu_count(), args.num_proc)
    )

    val_formatted = split_ds["test"].map(
        format_llama_nemotron_sample,
        remove_columns=split_ds["test"].column_names,
        num_proc=min(os.cpu_count(), args.num_proc)
    )

    os.makedirs(args.out_path, exist_ok=True)
    train_formatted.to_json(os.path.join(args.out_path, "training.jsonl"), num_proc=min(os.cpu_count(), args.num_proc))
    val_formatted.to_json(os.path.join(args.out_path, "validation.jsonl"), num_proc=min(os.cpu_count(), args.num_proc))
