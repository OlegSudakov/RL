import os

from datasets import load_dataset, concatenate_datasets

from nemo_rl.data.interfaces import TaskDataSpec


def format_llama_nemotron_sample(sample: dict[str, str]):
    messages = [{"role": "system", "content": sample["system_prompt"]}]
    messages += sample["input"]
    messages += [{"role": "assistant", "content": sample["output"]}]
    return {
        "messages": messages,
        "task_name": sample["category"]
    }


class LlamaNemotronDataset:
    def __init__(self, seed: int = 0, test_size: float = 0.01):
        original_ds = load_dataset("nvidia/Llama-Nemotron-Post-Training-Dataset", "SFT")
        original_ds = concatenate_datasets([dataset for dataset in original_ds.values()])
        split_ds = original_ds.train_test_split(test_size=test_size, seed=seed)

        train_formatted = split_ds["train"].map(
            format_llama_nemotron_sample,
            remove_columns=split_ds["train"].column_names,
            num_proc=min(os.cpu_count(), 32)
        )

        val_formatted = split_ds["test"].map(
            format_llama_nemotron_sample,
            remove_columns=split_ds["test"].column_names,
            num_proc=min(os.cpu_count(), 32)
        )

        self.formatted_ds = {
            "train": train_formatted,
            "validation": val_formatted
        }

        self.task_spec = TaskDataSpec(
            task_name="LlamaNemotron",
            prompt_file=None,
        )


class LlamaNemotronJsonlDataset:
    def __init__(self, train_ds_path: str, val_ds_path: str):
        train_dataset = load_dataset("json", data_files=train_ds_path)["train"]
        val_dataset = load_dataset("json", data_files=val_ds_path)["train"]

        self.formatted_ds = {
            "train": train_dataset,
            "validation": val_dataset,
        }

        self.task_spec = TaskDataSpec(
            task_name="LlamaNemotronJsonl",
            prompt_file=None,
        )
