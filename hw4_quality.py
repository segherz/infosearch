from hw4_BERT_search import get_BERT_index, corpus, names
from hw3_matrix_and_filenames_2 import get_corp, get_BM_25
import numpy as np


def main(questions_ind, answers_ind):
    """
    считает скалярное произведение между векторами вопросов и ответов
    на диагоналях оказались расстояния между соответствующими друг другу вопросами и ответами
    в каждой строке применяет функцию argsort()
    (предварительно поменять знак каждого элемента, чтобы сортировка получилась по убыванию)
    считает количество вопросов, для которых соотвтетсвующий ответ оказался в топ-5
    """
    prod = np.matmul(questions_ind, answers_ind.T)
    range_number_of_answer = np.diag(np.argsort(-prod, axis=1), 0)
    range_number_of_answer = range_number_of_answer < 6
    range_number_of_answer = range_number_of_answer.astype(int)
    return range_number_of_answer.sum() / range_number_of_answer.shape[0]

if __name__ == "__main__":
    BERT_answers_ind, BERT_questions_ind = get_BERT_index(corpus).numpy(), get_BERT_index(list(names)).numpy()
    print("BERT search quality: ", main(BERT_questions_ind, BERT_answers_ind))

    BM_25_answers_ind, BM_25_questions_ind = get_BM_25(corpus), get_BM_25(list(names))
    print("BM_25 search quality: ", main(BM_25_questions_ind, BM_25_answers_ind))

