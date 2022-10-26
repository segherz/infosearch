import numpy as np
import os
import scipy as sp
import json
from hw3_matrix_and_filenames import get_corp
from transformers import AutoTokenizer, AutoModel
import torch
from torch.nn.functional import normalize

DATA_NAME = 'data.jsonl'
PATH_TO_DATA = os.path.abspath(DATA_NAME)

tokenizer = AutoTokenizer.from_pretrained("sberbank-ai/sbert_large_nlu_ru")
model = AutoModel.from_pretrained("sberbank-ai/sbert_large_nlu_ru")

with open(PATH_TO_DATA, 'r') as f:
    texts = list(f)[:500]

corpus, names = get_corp(texts)

def get_BERT_index(sentences):
    encoded_input = tokenizer(sentences, padding=True, truncation=True, max_length=24, return_tensors='pt')
    attention_mask = encoded_input['attention_mask']

    with torch.no_grad():
        model_output = model(**encoded_input)

    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    return sum_embeddings / sum_mask


def main(corpus_ind, names):
    request = input("введите запрос: ")
    req_ind = get_BERT_index([request])
    names_mask = torch.argsort(torch.matmul(normalize(corpus_ind, dim=1), normalize(req_ind, dim=1).T), dim=-2)
    return names[names_mask.ravel()]

if __name__ == '__main__':
    corp_ind = get_BERT_index(corpus)
    accept = 'да'
    while accept == 'да':
        print(main(corp_ind, names))
        accept = input('Продолжить поиск? (да/нет): ')
