import json
import os
import random
from typing import Dict, Union

import numpy as np
import pandas as pd
import requests
import torch
import wandb
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast, TrainerCallback


class TextGenerationCallback(TrainerCallback):
    def __init__(self, test_dataset, tokenizer, model):
        self.test_dataset = test_dataset
        self.tokenizer = tokenizer
        self.model = model

        self.encoded_test_dataset = self.test_dataset.map(
            tokenize_and_encode, batched=True
        )
        self.encoded_test_dataset.set_format(
            type="torch", columns=["input_ids", "attention_mask"]
        )

        self.test_dataloader = DataLoader(
            self.encoded_test_dataset, batch_size=32, collate_fn=collate_test_batch
        )

    def on_evaluate(self, args, state, control):
        generated_texts = []
        with torch.no_grad():
            for batch in self.test_dataloader:
                input_ids = batch["input_ids"].to(self.model.device)
                attention_mask = batch["attention_mask"].to(self.model.device)
                generated = self.model.generate(
                    input_ids, attention_mask=attention_mask
                )
                generated_texts.extend(
                    self.tokenizer.batch_decode(generated, skip_special_tokens=True)
                )

        assert len(generated_texts) == self.test_dataset.shape[0]

        self.test_dataset["neutral_sentence"] = generated_texts

        response = requests.post(
            url="https://nlp-zh.skoltech.ru/clef2024/test",
            json={
                "toxic_sentence": self.test_dataset["toxic_sentence"].values.tolist(),
                "neutral_sentence": self.test_dataset[
                    "neutral_sentence"
                ].values.tolist(),
            },
            verify=False,
        )
        metrics = (
            pd.DataFrame(response.json())
            .groupby("lang")["J"]
            .mean()
            .round(3)
            .reindex(["en", "es", "de", "zh", "ar", "hi", "uk", "ru", "am"])
        )
        wandb.log({"metrics": wandb.Table(dataframe=metrics)})


def tokenize_and_encode(
    example: pd.Series,
    tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
) -> Union[Dict[str, torch.Tensor], pd.Series]:
    """
    Tokenizes and encodes the input sentences using the given tokenizer.

    Args:
        example (pd.Series): A pandas Series containing the input sentences.

    Returns:
        Union[Dict[str, torch.Tensor], pd.Series]: A dictionary or pandas Series with encoded input features.
    """
    encoded_text = tokenizer(
        [
            f"Detoxify text for {l}: {e}"
            for l, e in zip(example["lang"], example["toxic_sentence"])
        ],
        padding="max_length",
        max_length=128,
        truncation=True,
        return_tensors="pt",
    )

    try:
        encoded_label = tokenizer(
            example["neutral_sentence"],
            padding="max_length",
            truncation=True,
            max_length=128,
            return_tensors="pt",
            return_attention_mask=False,
            return_token_type_ids=False,
        )
        example["labels"] = encoded_label["input_ids"]
    except KeyError:
        pass

    example["input_ids"] = encoded_text["input_ids"]
    example["attention_mask"] = encoded_text["attention_mask"]
    return example


def collate_test_batch(batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    input_ids = [item["input_ids"] for item in batch]
    attention_mask = [item["attention_mask"] for item in batch]
    return {
        "input_ids": torch.stack(input_ids),
        "attention_mask": torch.stack(attention_mask),
    }


def load_config(config_file_path: str) -> Dict[str, Union[str, int, float]]:
    """Load configuration from a JSON file."""
    with open(config_file_path, "r") as f:
        config = json.load(f)
    return config


def set_random_seed(seed: int) -> None:
    """
    Set random seed for reproducibility across multiple runs.

    Args:
        seed (int): The seed value to set for random number generation.

    Returns:
        None: Only sets seed.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:2"
    os.environ["PL_GLOBAL_SEED"] = str(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
