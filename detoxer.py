import transformers
import torch
from transformers import pipeline

import pandas as pd
from tqdm import tqdm

tqdm.pandas()
torch.set_grad_enabled(False)

import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--language", help = "")
parser.add_argument("--nshot", type=int, help = "")
parser.add_argument("--model_path", help = "")
parser.add_argument("--hf_token", help = "")
parser.add_argument("--output_path", help = "")
parser.add_argument("--batch_size", type=int, help = "")

# Read arguments from command line
args = parser.parse_args()
lang = str(args.language)
model_path = args.model_path
N_SHOT = args.nshot
hf_token = args.hf_token

# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

tokenizer = AutoTokenizer.from_pretrained(model_path,      
    padding_side='left',                 
    token=hf_token,
    )

model = AutoModelForCausalLM.from_pretrained(model_path, 
    token=hf_token,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
    attn_implementation='eager',
    ).eval()

import torch
from transformers import pipeline

tokenizer.pad_token_id = tokenizer.eos_token_id

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device_map="auto",
    )

from datasets import load_dataset

dataset = load_dataset("textdetox/multilingual_paradetox")

#Bad word filter
def get_tokens_as_tuple(word):
    return tuple(tokenizer([word], add_special_tokens=False).input_ids[0])

stopwords = load_dataset("textdetox/multilingual_toxic_lexicon")
words = []
words.extend(stopwords[lang]["text"])
stopwords = set(words)
sequence_bias = {get_tokens_as_tuple(bad_word): -10.0 for bad_word in stopwords}

terminators = [
    pipe.tokenizer.eos_token_id,
    pipe.tokenizer.convert_tokens_to_ids("<|eot_id|>")
]

new_dataset = load_dataset("s-nlp/paradetox")

new_dataset = load_dataset("textdetox/multilingual_toxicity_dataset")

primed_data = []

if ((model_path.lower().startswith('mistral')) or (model_path.lower().find('phi-3')+1) or (model_path.lower().find('gemma')+1)) and not (model_path.lower().find('dolphin')+1):
    prompted_text = '''Your task is text style transfer. You rewrite the text into non-toxic language. You must match the target style and preserve the original meaning as much as possible. You should not need to explain the response. You cannot hallucinate or add anything outside the original input text. You should not include the input text in the response. You should only generate the target text. \n'''
    end_text = ""
    dataset_to_eval = dataset[lang][-N_SHOT:].values() if N_SHOT>0 else dataset[lang][:N_SHOT].values()

    for toxic_sentence, neutral_sentence in zip(*dataset_to_eval):
        end_text += "\n"+"Toxic text: "+toxic_sentence#+"\n"
        end_text += "\n"+"Neutral text: "+neutral_sentence#+"\n"
    if len(end_text) > 1:
        end_text = '\nHere are some examples:\n'+end_text
    prompted_text = prompted_text+end_text+"\nToxic text: "
    for i in new_dataset['train']['en_toxic_comment']:
        messages = [
        {"role": "user", 
        "content":prompted_text+i+"\nNeutral text: "
            },
        ]

        prompt = pipe.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
        )

        primed_data.append(prompt)

else:
    def prompt_maker(lang, k=5):
        end_text = []
        if k == 0:
            return end_text
        for toxic_sentence, neutral_sentence in zip(*dataset[lang][-k:].values()):
            end_text.extend([{"role": "user", "content": toxic_sentence }])
            end_text.extend([{"role": "assistant", "content": neutral_sentence }])

        return end_text
    for i in new_dataset[lang]['text']:
    
        messages = [
        {"role": "system", 
        "content":"Your task is text style transfer. You rewrite the text into non-toxic language. You must match the target style and preserve the original meaning as much as possible. You should not need to explain the response. You cannot hallucinate or add anything outside the original input text. You should not include the input text in the response. You should only generate the target text. "},
        *prompt_maker(lang, k=N_SHOT),
        {"role": "user", "content": i},
        ]
        prompt = pipe.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
        )
        primed_data.append(prompt)

print('='*20)
print(primed_data[0])
print('='*20)

def data():
    for i in tqdm(primed_data):
        yield i

new = []

for out in pipe(
        data(),
        max_new_tokens=64,
        add_special_tokens = True,
        temperature=0.8, top_p=0.9, do_sample=True,
        batch_size=args.batch_size,
        return_full_text=False,
    ):
    new.append(out[0]['generated_text'])

output = pd.DataFrame(new_dataset[lang])
output['generated_neutral_sentence'] = new
output.to_csv(args.output_path)