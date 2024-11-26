import torch
from typing import Dict, Union, List
import pandas as pd
from transformers import (
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
    AutoModelForSeq2SeqLM,
)
from tqdm.auto import tqdm
from sklearn.model_selection import train_test_split


def translate_batch(
    texts: List[str],
    model: AutoModelForSeq2SeqLM,
    tokenizer: PreTrainedTokenizerFast,
    batch_size: int = 32,
    verbose: bool = True,
    src_lang: str = "eng_Latn",
    tgt_lang: str = "rus_Cyrl",
) -> List[str]:
    """
    Translate a batch of texts.

    Args:
        texts (List[str]): The list of texts to translate.
        model (M2M100ForConditionalGeneration): The translation model.
        tokenizer (PreTrainedTokenizerFast): The tokenizer for the translation model.
        batch_size (int, optional): The batch size for translation. Defaults to 32.
        verbose (bool, optional): Toggle translation progressbar
        src_lang (str, optional): Source language code.
        tgt_lang (str, optional): Target language code.
    Returns:
        List[str]: The translated texts.
    """
    translations = []

    tokenizer.src_lang = src_lang
    tokenizer.tgt_lang = tgt_lang

    if verbose:
        iterator = tqdm(range(0, len(texts), batch_size), desc="Translating")
    else:
        iterator = range(0, len(texts), batch_size)

    for i in iterator:
        batch = texts[i : i + batch_size]
        tokenized_batch = tokenizer(
            batch, return_tensors="pt", padding=True, truncation=True
        )
        input_ids = tokenized_batch.input_ids.to(model.device)
        attention_mask = tokenized_batch.attention_mask.to(model.device)
        batch_translated = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            forced_bos_token_id=tokenizer.lang_code_to_id[tokenizer.tgt_lang],
        )
        translations.extend(
            tokenizer.decode(
                tokens, skip_special_tokens=True, clean_up_tokenization_spaces=True
            )
            for tokens in batch_translated
        )
    return translations


def split_by_language(
    data: pd.DataFrame, random_seed: int = 42, test_size: float = 0.1
) -> Union[pd.DataFrame, pd.DataFrame]:

    train_data = []
    dev_data = []

    for lang in data.lang.unique():
        lang_data = data[data["lang"] == lang]
        lang_train, lang_dev = train_test_split(
            lang_data, test_size=test_size, random_state=random_seed
        )
        train_data.append(lang_train)
        dev_data.append(lang_dev)

    train_data = pd.concat(train_data)
    dev_data = pd.concat(dev_data)

    return train_data, dev_data
