import numpy as np
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
import pymorphy2
import os
import scipy as sp
import json 

DATA_NAME = 'data.jsonl'
PATH_TO_DATA = os.path.abspath(DATA_NAME)
STOPS_RUS = set(stopwords.words('russian'))
k = 2
b = 0.75
tokenizer = RegexpTokenizer(r'[а-яА-ЯёЁ]+')
tfidf_vectorizer = TfidfVectorizer(use_idf=True, norm='l2', stop_words=STOPS_RUS)
count_vectorizer = CountVectorizer()
morph = pymorphy2.MorphAnalyzer()


def preprocess(text):
    """
    разбивает текст на токены и приводит к начальной форме, исключает стоп-слова
    """
    text = text.lower()
    tokens_list = tokenizer.tokenize(text)
    res = []
    for token in tokens_list:
        normalized_token = morph.parse(token)[0].normal_form
        if normalized_token not in STOPS_RUS and normalized_token not in '0123456789':
            res.append(normalized_token)
    return ' '.join(res)


def return_answer(question):
    """
    получает на вход распарсенный json
    если у вопроса нет ответов, возвращает пустую строку
    получает список вида (текст ответа, value)
    сортирует его по value и возвращает текст самого последнего элемента
    """
    if not question['answers']:
        return ''

    return sorted(
        list(
            map(lambda x: [x['text'], x['author_rating']['value']] if x['author_rating']['value']
                else [x['text'], 0],
                question['answers'])
        ), key=lambda x: int(x[-1])
    )[-1][0]


def get_corp(corp):
    """
    собирает корпус
    """
    corpus = []
    doc_names = []

    for mail_ru_question in corp:
        parsed_mail_ru_question = json.loads(mail_ru_question)
        answer_text = return_answer(parsed_mail_ru_question)
        processed_text = preprocess(answer_text)
        corpus.append(processed_text)
        doc_names.append(parsed_mail_ru_question['question'])
    return corpus, np.array(doc_names)


def get_BM_25(corpus):
    """
    создает doc-term матрицу tf для корпуса, вычисляет знаминатель BM-25
    затем обучает tfidf-векторайзер на корпусе, получает idf, вычисляет числитель
    получает результирующую матрицу для BM-25
    """
    tf = count_vectorizer.fit_transform(corpus)
    len_docs = tf.sum(axis=1)
    avgdl = len_docs.mean()
    denominator_length = (k * (1 - b + b * len_docs / avgdl))
    denominator = tf + denominator_length

    tfidf_vectorizer.fit(corpus)
    idf = tfidf_vectorizer.idf_
    idf = sp.sparse.csr_matrix(np.expand_dims(idf, axis=0))
    numerator = idf.multiply(tf)  # почему-то с dot плохо работало

    res_mat = numerator / denominator

    return res_mat


def initialize_corp_and_matrix():
    """
    инициализирует корпус и названия документов для импорта в следующую функцию
    """
    with open(PATH_TO_DATA, 'r') as f:
        texts = list(f)[:50000]

    corpus, names = get_corp(texts)
    BM_25 = get_BM_25(corpus)

    return BM_25, names
