import pandas as pd
import logging
import argparse
import wandb
from datasets import Dataset
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
)
from utils.training_utils import load_config, set_random_seed
import os
from sklearn.model_selection import train_test_split
import torch
from typing import Dict, Union

if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config_path",
        type=str,
        required=True,
        help="Path to a '.json' config for fine-tuning",
    )

    parser.add_argument(
        "--train_data_path", required=False, default="data/dev_with_answers.tsv"
    )

    parser.add_argument(
        "--translated_data_path",
        required=False,
        default="data/translated_paradetox_joint.csv",
    )

    parser.add_argument(
        "--input_columns",
        required=False,
        default="toxic_text",
    )

    parser.add_argument(
        "--output_columns",
        required=False,
        default="neutral_text",
    )
    parser.add_argument(
        "--test_data_path", required=False, default="data/test_without_answers.tsv"
    )

    args = parser.parse_args()

    config = load_config(args.config_path)


    wandb.require("core")
    
    # config["train_args"]["output_dir"] += f"{config['model_args']['model_name']}"

    logging.info(f"Setting random seed to {config['train_args']['seed']}")
    set_random_seed(config["train_args"]["seed"])

    logging.info("Loading model.")
    tokenizer = AutoTokenizer.from_pretrained(
        config["model_args"]["model_name"], legacy=False
    )
    model = AutoModelForSeq2SeqLM.from_pretrained(
        config["model_args"]["model_name"]
    ).cuda()

    logging.info(f"Loading training data: '{config['data_args']['train_data_path']}'")

    train_data_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        config["data_args"]["train_data_path"],
    )
    logging.debug(train_data_path)
    data = pd.read_csv(train_data_path)

    #data["generated_neutral_sentence"] = data['generated_neutral_sentence'].apply(lambda a: a.split('\n')[-1])
    #data['flag_stop'] = data['generated_neutral_sentence']\
    #    .apply(lambda a: any([a.lower().startswith(bad_ans) for bad_ans in \
    #                            ['i cannot', "i can't", "i can not" ,'i understand',
    #                            'i strongly', "i apologize", "i condemn","i'm deeply"] ]) )
    #print(data['flag_stop'].value_counts())
    #data = data[data['is_toxic'] == 1]
    #data = data[data['en_len'] > 10]
    #data = data[['en','generated_toxic_sentence']]
    #print(data)

    if config["data_args"]["use_translated_paradetox"]:
        assert (
            "translated_data_path" in config["data_args"]
        ), "translated_data_path is required when use_translated_paradetox is True"

        logging.info(
            f"""Using translated paradetox, loaded from: \
                     {config["data_args"]["translated_data_path"]}"""
        )

        trans_data_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            config["data_args"]["translated_data_path"],
        )
        trans_data = pd.read_csv(trans_data_path)

        data = pd.concat([data, trans_data], ignore_index=True, axis=0)

    logging.info("Splitting to train/validation")
    train_data, eval_data = train_test_split(
        data,
        random_state=config["train_args"]["seed"],
        test_size=config["data_args"]["eval_size"],
    )

    train_dataset = Dataset.from_pandas(train_data)
    eval_dataset = Dataset.from_pandas(eval_data)

    logging.info("Preprocessing train data")

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
            example[args.input_columns],
            padding="max_length",
            max_length=256,
            truncation=True,
            return_tensors="pt",
        )

        try:
            encoded_label = tokenizer(
                example[args.output_columns],
                padding="max_length",
                truncation=True,
                max_length=256,
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

    tokenize_and_encode_wrapper = lambda x: tokenize_and_encode(x, tokenizer=tokenizer)
    encoded_train_dataset = train_dataset.map(tokenize_and_encode_wrapper, batched=True)
    encoded_train_dataset.set_format(
        type="torch", columns=["input_ids", "attention_mask", "labels"]
    )

    logging.info("Preprocessing eval data")
    encoded_dev_dataset = eval_dataset.map(tokenize_and_encode_wrapper, batched=True)
    encoded_dev_dataset.set_format(
        type="torch", columns=["input_ids", "attention_mask", "labels"]
    )

    if config["train_args"]["do_predict"]:
        assert (
            "test_data_path" in config["data_args"]
        ), "'do_predict' set to True, path to the test data should be provided."

        test_data_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            config["data_args"]["test_data_path"],
        )
        test_dataset = Dataset.from_pandas(
            pd.read_csv(
                test_data_path, sep="\t" if test_data_path.endswith(".tsv") else "None"
            )
        )

    train_args = Seq2SeqTrainingArguments(**config["train_args"])

    label_pad_token_id = tokenizer.pad_token_id
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=label_pad_token_id,
        pad_to_multiple_of=8 if train_args.fp16 else None,
    )

    trainer = Seq2SeqTrainer(
        args=train_args,
        model=model,
        tokenizer=tokenizer,
        # compute_metrics=lambda x: compute_chrf(x, tokenizer=tokenizer) if train_args.predict_with_generate else None,
        train_dataset=encoded_train_dataset,
        eval_dataset=encoded_dev_dataset,
        data_collator=data_collator,
    )

    wandb.require("core")
    wandb.init(name=train_args.run_name, config=config)
    wandb.watch(model)

    trainer.train()
