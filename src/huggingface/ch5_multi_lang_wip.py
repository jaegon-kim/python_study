from datasets import get_dataset_config_names
from datasets import load_dataset
from datasets import DatasetDict
from collections import defaultdict 

import pandas as pd

import torch
import torch.nn as nn
from transformers import AutoTokenizer
from transformers import XLMRobertaConfig
from transformers import XLMRobertaForTokenClassification
from transformers.modeling_outputs import TokenClassifierOutput
from transformers.models.roberta.modeling_roberta import RobertaModel
from transformers.models.roberta.modeling_roberta import RobertaPreTrainedModel




def construct_panx_ch():

    # 스위치 말뭉치
    langs = ["de", "fr", "it", "en"]
    fracs = [0.629, 0.229, 0.084, 0.059]

    panx_ch = defaultdict(DatasetDict)

    for lang, frac in zip(langs, fracs):
        # Loading 
        ds = load_dataset("xtreme", name=f"PAN-X.{lang}")
        for split in ds:
            panx_ch[lang][split] = (ds[split]
                                    .shuffle(seed=0)
                                    .select(range(int(frac * ds[split].num_rows))))
    return panx_ch    

def test_panx_deutsch(panx_ch):
            
    print(pd.DataFrame(
        {lang: [panx_ch[lang]["train"].num_rows] for lang in langs},
        index=["Number of tranining examples"])        
    )

    element = panx_ch["de"]["train"][0]
    print(element)

    print(element.items())
    print()
    for key, value in element.items():
        print(f"{key}: {value}")

    for key, value in panx_ch["de"]["train"].features.items():
        print(f"{key}: {value}")

    tags = panx_ch["de"]["train"].features["ner_tags"].feature
    print(tags)

    print('* original')
    print(panx_ch["de"]["train"][0])

    def create_tag_names(batch):
        return {"ner_tags_str": [tags.int2str(idx) for idx in batch["ner_tags"]]}

    panx_de = panx_ch["de"].map(create_tag_names)
    de_example = panx_de["train"][0]

    print('\n* ner_tags_str is added')
    print(de_example)

    print()
    for key, value in de_example.items():
        print(f"{key}: {value}")

    print(pd.DataFrame([de_example["tokens"], de_example["ner_tags_str"]], index=['Tokens', 'Tags']))

    from collections import Counter

    split2freqs = defaultdict(Counter)

    for split, dataset in panx_de.items():
        for row in dataset["ner_tags_str"]:
            for tag in row:
                if tag.startswith("B"):
                    tag_type = tag.split("-")[1]
                    split2freqs[split][tag_type] += 1

    print(
        pd.DataFrame.from_dict(split2freqs, orient="index")
    )


panx_ch = construct_panx_ch()
#test_panx_deutsch(panx_ch)

xlmr_model_name = "xlm-roberta-base"
xlmr_tokenizer = AutoTokenizer.from_pretrained(xlmr_model_name)

tags = panx_ch["de"]["train"].features["ner_tags"].feature
print(tags)

def create_tag_names(batch):
    return {"ner_tags_str": [tags.int2str(idx) for idx in batch["ner_tags"]]}

panx_de = panx_ch["de"].map(create_tag_names)


def tokenize_and_align_labels(examples):
    tokenized_inputs = xlmr_tokenizer(examples["tokens"], truncation=True, is_split_into_words=True)
    labels = []
    for idx, label in enumerate(examples["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=idx)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None or word_idx == previous_word_idx:
                label_ids.append(-100)
            else:
                label_ids.append(label[word_idx])
            previous_word_idx = word_idx
        labels.append(label_ids)
    tokenized_inputs["labels"] = labels
    return tokenized_inputs

def encode_panx_dataset(corpus):
    return corpus.map(tokenize_and_align_labels, batched=True, remove_columns=['langs', 'ner_tags', 'tokens'])

panx_de_encoded = encode_panx_dataset(panx_ch["de"])
