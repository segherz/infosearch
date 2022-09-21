from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
import os
from sklearn.feature_extraction.text import TfidfVectorizer
import pymorphy2
import numpy as np

STOPS_RUS = set(stopwords.words('russian'))
tokenizer = RegexpTokenizer(r'[а-яА-ЯёЁ]+')
vectorizer = TfidfVectorizer(stop_words=STOPS_RUS)
PATH_TO_FRIENDS_FOLDER = 'friends-data'
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


def get_tfidf_vecs(path):
    """
    собирает корпус и веторизует его
    """
    corpus = []
    doc_names = []
    for root, dirs, files in os.walk(path):
        if '.DS_Store' in files:
            continue
        for name in files:
            if name != '.DS_Store':
                filepath = os.path.join(root, name)
                with open(filepath, 'r', encoding='utf-8') as file:
                    filetext = file.read().lower()
                    text = preprocess(filetext)
                    corpus.append(text)
                doc_names.append(name)
    mat = vectorizer.fit_transform(corpus)
    return mat, doc_names


def save_matrix_filenames(matrix, filenames):
    with open('friends_tfidf_mat_and_names.npy', 'wb') as f:
        np.save(f, matrix.toarray())
        np.save(f, np.array(filenames))

def main(path):
    matr_ind, doc_names = get_tfidf_vecs(path)
    save_matrix_filenames(matr_ind, doc_names)

main(PATH_TO_FRIENDS_FOLDER)