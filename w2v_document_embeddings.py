import gensim.downloader
import gensim
import pandas as pd
import os
import pickle
import numpy as np
from textProcessing import tokenize


def read_document(file_path):
    return pd.read_csv(file_path, dtype=str, usecols=[1, 2])


def embed_document(model, tokens):
    total_vector = np.zeros([model.vector_size], dtype=np.float32)
    normalization = 0
    for token in tokens:
        if token in model:
            embedding = model[token]
            total_vector += embedding
            normalization += 1
    if normalization == 0:
        return total_vector
    return total_vector/normalization


def embed_documents(model, posts):
    embedded_posts = list()
    for i, post in enumerate(posts):
        print(100*i/len(posts))
        tokens = tokenize(post)
        embedded_post = embed_document(model, tokens)
        embedded_posts.append(embedded_post)
    embedded_posts.reverse()
    return embedded_posts


intermediate_location = f"{os.path.basename(__file__)}-intermediate_data"
if not os.path.exists(intermediate_location):
    os.mkdir(intermediate_location)

print(intermediate_location)

if __name__ == '__main__':
    print("Loading word2vec...")
    model_gn: gensim.models.Word2Vec = gensim.downloader.load('word2vec-google-news-300')
    print("loaded")

    data_path = 'data'
    file_name = 'jigsaw-toxic.csv'
    file_path = os.path.join(data_path, file_name)
    data = read_document(file_path)
    posts = list(data['text'])
    labels = list(data['class'])

    embedding_file_path = os.path.join(intermediate_location, f"{file_name}-embeddings.p")

    post_embeddings = embed_documents(model_gn, posts)
    with open(embedding_file_path, "wb") as f:
        pickle.dump(post_embeddings, f)

    # tokens = tokenize(data)
    # embedding = model_gn[tokens[0]]
    #
    # print(model_gn.most_similar(tokens[0]))
    # print(model_gn.similar_by_vector(embedding))

