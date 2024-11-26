import argparse
import logging
import os

import pandas as pd
from datasets import load_dataset
from tqdm.auto import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from utils.data_utils import translate_batch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

LANG_ID_MAPPING = {
    "ru": "rus_Cyrl",
    "en": "eng_Latn",
    "am": "amh_Ethi",
    "es": "spa_Latn",
    "uk": "ukr_Cyrl",
    "zh": "zho_Hans",
    "ar": "arb_Arab",
    "hi": "hin_Deva",
    "de": "deu_Latn",
}

model_map = {
    "small": "facebook/nllb-200-distilled-600M",
    "medium": "facebook/nllb-200-1.3B",
    "large": "facebook/nllb-200-3.3B",
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--nllb_size",
        required=False,
        help="NLLB translation model version. Options are 'small', 'medium' and 'large'.",
        default="large",
        choices=["small", "medium", "large"],
    )
    parser.add_argument(
        "--batch_size",
        required=False,
        default=64,
        type=int,
        help="Batch size for translation. Default is 64.",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Whether to display stuff", default=True
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Where to save the translated data.",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    logger.info("Loading models")
    model = (
        AutoModelForSeq2SeqLM.from_pretrained(model_map[args.nllb_size]).cuda().eval()
    )
    tokenizer = AutoTokenizer.from_pretrained(model_map[args.nllb_size])

    logger.info("Loading dataset")
    dataset = load_dataset("s-nlp/paradetox", split="train")

    joint_df = pd.DataFrame(
        {
            "toxic_comment": dataset["en_toxic_comment"],
            "neutral_comment": dataset["en_neutral_comment"],
            "language": ["en"] * len(dataset),
        }
    )

    logger.info("Translating ParaDetox")
    translated_dfs = []

    for lang in tqdm(
        LANG_ID_MAPPING.keys(), desc=f"Translating", disable=not args.verbose
    ):

        translated_toxic = translate_batch(
            texts=dataset["en_toxic_comment"],
            model=model,
            tokenizer=tokenizer,
            batch_size=args.batch_size,
            verbose=args.verbose,
            src_lang=LANG_ID_MAPPING["en"],
            tgt_lang=LANG_ID_MAPPING[lang],
        )
        translated_neutral = translate_batch(
            texts=dataset["en_neutral_comment"],
            model=model,
            tokenizer=tokenizer,
            batch_size=args.batch_size,
            verbose=args.verbose,
            src_lang=LANG_ID_MAPPING["en"],
            tgt_lang=LANG_ID_MAPPING[lang],
        )

        df = pd.DataFrame(
            {
                "toxic_comment": translated_toxic,
                "neutral_comment": translated_neutral,
                "language": [lang] * len(translated_neutral),
            }
        )

        translated_dfs.append(df)

    joint_df = pd.concat([joint_df] + translated_dfs, axis=0, ignore_index=True)

    output_file = f"{args.output_dir}/translated_paradetox_joint.csv"
    joint_df.to_csv(output_file, index=False)
    logging.info(f"Translated data saved to '{output_file}'")
