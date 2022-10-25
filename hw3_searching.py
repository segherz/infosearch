from hw3_matrix_and_filenames import count_vectorizer, preprocess, initialize_corp_and_matrix
import numpy as np

BM_25, names = initialize_corp_and_matrix()

def main(BM_25, filenames):
    query = input('Введите запрос: ')
    query_prepared = preprocess(query)
    query_count_vec = count_vectorizer.transform([query_prepared])
    freqs = BM_25 * query_count_vec.T
    sorted_freqs = np.argsort(freqs, axis=0)[::-1]
    return names[sorted_freqs.ravel()]

if __name__ == '__main__':
    accept = 'да'
    while accept == 'да':
        print(main(BM_25, names))
        accept = input('Продолжить поиск? (да/нет): ')
