from hw2_saving_matrix_and_filenames import preprocess, vectorizer
from pprint import pprint
import numpy as np

with open('friends_tfidf_mat_and_names.npy', 'rb') as f:
    doc_term_matrix = np.load(f, allow_pickle=True)
    doc_names = np.load(f, allow_pickle=True)

def indexing_request(req_text):
    preprocessed_req_text = preprocess(req_text)
    return vectorizer.transform([preprocessed_req_text]).toarray()[0]

def find_cosine_similarities(indices_mat, req_array):
    """
    вычисляет косинусную близость для вектора запроса и каждой строки матрицы
    """
    similarity_scores = []
    c = -1
    for doc_vector in indices_mat:
        c += 1
        similarity_scores.append((np.dot(doc_vector, req_array) / (np.linalg.norm(doc_vector)*np.linalg.norm(req_array)), doc_names[c]))
    return similarity_scores

def main(request):
    request_index = indexing_request(request)
    cosine_similarities = find_cosine_similarities(doc_term_matrix, request_index)
    cosine_similarities.sort(key = lambda x: x[0], reverse = True)
    return [result[1] for result in cosine_similarities]

request = input('Введите запрос: ')

pprint(main(request))