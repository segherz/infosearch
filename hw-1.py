import os
from nltk.corpus import stopwords
import pymorphy2
from nltk.tokenize import RegexpTokenizer
from sklearn.feature_extraction.text import CountVectorizer
from math import inf
import numpy as np
from collections import defaultdict

import nltk
nltk.download('stopwords')

vectorizer = CountVectorizer(analyzer='word')
tokenizer = RegexpTokenizer(r'[а-яА-ЯёЁ]+')
morph = pymorphy2.MorphAnalyzer()

STOPS_RUS = set(stopwords.words('russian'))
PATH_TO_FRIENDS_FOLDER = 'friends-data'
CHARACTERS = {
    'Моника': ['моника', 'мон'],
    'Рэйчел': ['рэйчел', 'рейч'],
    'Чендлер': ['чендлер', 'чэндлер', 'чен'],
    'Фиби': ['фиби', 'фибс'],
    'Росс': ['росс'],
    'Джоуи': ['джоуи', 'джои', 'джо']
}
NAMES = {name: character for character in CHARACTERS.keys() for name in CHARACTERS[character]}


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


def get_reverse_indices(path):
    """
    собирает корпус и веторизует его
    """
    corpus = []
    for root, dirs, files in os.walk(path):
        if '.DS_Store' in files:
            continue
        for name in files:
            if name != '.DS_Store':
                filepath = os.path.join(root, name)
                with open(filepath, 'r', encoding='utf-8') as file:
                    filetext = file.read()
                    text = preprocess(filetext)
                    corpus.append(text)
    return vectorizer.fit_transform(corpus)


def get_freq_mat(indices):
    """
    получает из обратного индекса матрицу суммарных частот для слов
    добавляет к частотам сами слова
    транспонирует, чтобы в каждой строке матрицы оказалось одно слово и его частота
    """
    words_freq = np.asarray(indices.sum(axis=0)).ravel()
    matrix_freq = np.array([np.array(vectorizer.get_feature_names_out()), words_freq])
    matrix_freq = np.array(np.transpose(matrix_freq))
    return matrix_freq


def get_least_and_most_freq_words(freq_matrix):
    min_freq, max_freq = inf, 0
    list_of_min_freqs, list_of_max_freqs = [], []
    for word, freq in freq_matrix:
        word = str(word)
        freq = int(freq)
        if freq <= min_freq:
            if freq < min_freq:
                list_of_min_freqs = []
                min_freq = freq
            list_of_min_freqs.append(word)

        if freq >= max_freq:
            if freq > max_freq:
                list_of_max_freqs = []
                max_freq = freq
            list_of_max_freqs.append(word)
    return list_of_min_freqs, list_of_max_freqs


def get_most_popular_character(freq_matrix):
    """
    считает количество вхождений в корпус для каждого персонажа
    затем ищет самого частотного среди них
    """
    names_count = defaultdict(int)
    for word, freq in freq_matrix:
        if word in NAMES.keys():
            names_count[NAMES[word]] += 1

    cnt = 0
    for character in names_count:
        if cnt < names_count[character]:
            most_freq_char = character
            cnt = names_count[character]
    return most_freq_char


def get_common_words_for_all_docs(docs_term):
    """
    транспонирует матрицу обратных индексов
    для вектора частот каждого слова считает кол-во документов с нулем появлений токена
    возвращает все слова, где нет ни одного "нулевого" документа
    """
    words_in_all_docs = []
    term_docs = np.transpose(docs_term).toarray()
    for i in range(term_docs.shape[0]):
        word_freqs_arrow = term_docs[i]
        count_docs_without_word = np.sum(word_freqs_arrow == 0)
        if count_docs_without_word == 0:
            words_in_all_docs.append(vectorizer.get_feature_names_out()[i])
    return words_in_all_docs


def main():
    reverse_indices = get_reverse_indices(PATH_TO_FRIENDS_FOLDER)
    freq_matrix = get_freq_mat(reverse_indices)

    least_freq, most_freq = get_least_and_most_freq_words(freq_matrix)
    most_popular_character = get_most_popular_character(freq_matrix)
    common_words = get_common_words_for_all_docs(reverse_indices)

    return most_popular_character, least_freq, most_freq, common_words


if __name__ == '__main__':
    main()
