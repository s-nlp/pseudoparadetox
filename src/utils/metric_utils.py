from typing import Dict, Union

import numpy as np
import wandb
from sacrebleu import CHRF
from transformers import EvalPrediction, PreTrainedTokenizer, PreTrainedTokenizerFast
from typing import List
from sentence_transformers import SentenceTransformer
from sentence_transforrmes.util import cos_sim

def compute_chrf(
    p: EvalPrediction, tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast]
) -> Dict[str, float]:
    """
    Compute the character-level F-score (CHRF) between model predictions and labels.

    Args:
        p (EvalPrediction): Evaluation predictions containing model predictions and labels.
        tokenizer (Union[PretrainedTokenizer, PretrainedTokenizerFast]): Tokenizer for decoding predictions and labels.

    Returns:
        Dict[str, float]: A dictionary containing the computed CHRF score.
    """
    chrf = CHRF()
    predictions, labels = p
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    chrf_scores = np.array(
        [
            chrf.sentence_score(hypothesis, [reference]).score / 100
            for hypothesis, reference in zip(decoded_preds, decoded_labels)
        ],
        dtype=np.float64,
    )
    wandb.log({"CHRF score": np.mean(chrf_scores)})
    return {"chrf_score": np.mean(chrf_scores)}

def get_sim_score(
        model: SentenceTransformer,
        source_texts: List[str],
        generated_texts: List[str],
        batch_size: int=128,
        normalize_embeddings: bool=True,
        verbose: bool=True
        ):

    """
    Calculate the cosine similarity between two lists of text strings using a pre-trained SentenceTransformer model.

    Args:
        model (SentenceTransformer): A pre-trained SentenceTransformer model.
        source_texts (List[str]): A list of source text strings.
        generated_texts (List[str]): A list of generated text strings.
        batch_size (int, optional): The batch size to use when encoding the text. Defaults to 128.
        normalize_embeddings (bool, optional): Whether to normalize the embeddings to unit length. Defaults to True.
        verbose (bool, optional): Whether to show a progress bar when encoding the text. Defaults to True.

    Returns:
        List[float]: A list of cosine similarity scores between the source and generated texts.


    Example:
        >>> from sentence_transformers import SentenceTransformer
        >>> model = SentenceTransformer('Alibaba-NLP/gte-large-en-v1.5', trust_remote_code=True)
        >>> source_texts = ['This is a positive sentence.', 'This is a negative sentence.']
        >>> generated_texts = ['This is a very positive sentence.', 'This is a somewhat negative sentence.']
        >>> sim_scores = get_sim_score(model, source_texts, generated_texts)
        >>> print(sim_scores)
        [0.97851303, 0.93650627]
    """
    if len(source_texts) != len(generated_texts):
        raise ValueError("The source_texts and generated_texts lists must have the same length.")


    source_embs = model.encode(
        sentences=source_texts,
        batch_size=batch_size,
        normalize_embeddings=normalize_embeddings,
        show_progress_bar=verbose,
        convert_to_tensor=False
        )   
    
    generated_embs = model.encode(
        sentences=generated_texts,
        batch_size=batch_size,
        normalize_embeddings=normalize_embeddings,
        show_progress_bar=verbose,
        convert_to_tensor=False
        )   

    sim_scores = []
    for source_emb, generated_emb in zip(source_embs, generated_embs):
        sim_score = cos_sim(source_emb, generated_emb)
        sim_scores.append(sim_score.item())

    return sim_scores
