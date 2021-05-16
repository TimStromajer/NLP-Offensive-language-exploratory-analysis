import gensim.downloader
import gensim
import pandas as pd
import os
import pickle
import numpy as np

from lazy_load import lazy
from scipy.spatial import distance
from textProcessing import tokenize, remove_links, remove_ats, remove_consecutive_phrases
from speech_classes import SPEECH_CLASSES
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from statistics import mean
from sklearn.manifold import TSNE


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
        post = remove_links(post)
        post = remove_ats(post)
        tokens = tokenize(post)
        tokens = remove_consecutive_phrases(tokens)
        embedded_post = embed_document(model, tokens)
        embedded_posts.append(embedded_post)
    embedded_posts.reverse()
    return embedded_posts


def load_or_create(path, action, recreate=False):
    if not recreate and os.path.exists(path):
        with open(path, "rb") as f:
            print(f"Loaded from {path}")
            output = pickle.load(f)
    else:
        output = action()
        with open(path, "wb") as f:
            pickle.dump(output, f)
            print(f"Generated and saved to {path}")
    return output


# def plot_similar_words(title, labels, embedding_clusters, filename=None):
#     plt.figure(figsize=(16, 9))
#     colors = cm.rainbow(np.linspace(0, 1, len(labels)))
#     for label, embeddings, color in zip(labels, embedding_clusters, colors):
#         x = embeddings[:, 0]
#         y = embeddings[:, 1]
#         plt.scatter(x, y, c=color, alpha=0.7, label=label)
#         plt.annotate(label.upper(), alpha=1.0, xy=(mean(x), mean(y)), xytext=(0, 0),
#                      textcoords='offset points', ha='center', va='center', size=15)
#     plt.legend(loc=4)
#     plt.title(title)
#     plt.grid(False)
#     if filename:
#         plt.savefig(filename, format='png', dpi=150, bbox_inches='tight')
#     plt.show()
#
#
# def plotPCA(title, labels, embedding_clusters, filename = None):
#     lens = [len(x) for x in embedding_clusters]
#     print("Creating array")
#     combined_embeddings = np.empty((sum(lens), len(embedding_clusters[0][0])), dtype=np.float32)
#     index = 0
#     print("Filling array")
#     for c in embedding_clusters:
#         for embedding in c:
#             combined_embeddings[index, :] = embedding
#         index += 1
#     print("Fitting PCA")
#     #model_en_2d = PCA(n_components=2, random_state = 32)
#     model_en_2d = TSNE(perplexity=15, n_components=2, init='pca', n_iter=3500, random_state=32)
#     model_en_2d = model_en_2d.fit_transform(combined_embeddings)
#     model_en_2d = np.array(model_en_2d)
#
#     combined_index = 0
#     mapped_embeddings = list()
#     for label_count in lens:
#         label_array = np.empty((label_count, 2), dtype=np.float32)
#         for i in range(label_count):
#             label_array[i, :] = model_en_2d[combined_index, :]
#             combined_index += 1
#         mapped_embeddings.append(label_array)
#     plot_similar_words(title, labels, mapped_embeddings, filename)
#
#
intermediate_location = f"{os.path.basename(__file__)}-intermediate_data"
if not os.path.exists(intermediate_location):
    os.mkdir(intermediate_location)


if __name__ == '__main__':
    regenerate = True
    def load_w2v():
        print("Loading word2vec...")
        w2v_model = gensim.downloader.load('word2vec-google-news-300')
        print("Loaded.")
        return w2v_model
    model = lazy(load_w2v)
    input_files = ['9.csv',
                   '21.csv',
                   '25.csv',
                   '26.csv',
                   '31.csv',
                   '32.csv',
                   'jigsaw-toxic.csv'
                   ]

    all_posts = list()
    all_labels = list()
    all_embeddings = list()

    for file_name in input_files:
        data_path = 'data'
        file_path = os.path.join(data_path, file_name)
        data = read_document(file_path)
        posts = list(data['text'])
        labels = list(data['class'])

        embedding_file_path = os.path.join(intermediate_location, f"{file_name}-embeddings.p")
        post_embeddings = load_or_create(embedding_file_path, lambda: embed_documents(model, posts), recreate=regenerate)

        all_posts.extend(posts)
        all_labels.extend(labels)
        all_embeddings.extend(post_embeddings)
        print(len(post_embeddings))

    labels_embeddings = dict()
    label_posts = dict()
    for i in range(len(all_labels)):
        label = all_labels[i]
        embedding = all_embeddings[i]
        if label not in labels_embeddings:
            labels_embeddings[label] = list()
            label_posts[label] = list()
        labels_embeddings[label].append(np.array(all_embeddings[i]))
        label_posts[label].append(all_posts[i])

    for label in labels_embeddings:
        if SPEECH_CLASSES[int(label)] == 'none':
            continue
        combined = sum(labels_embeddings[label])
        count = len(labels_embeddings[label])
        centroid = combined/count
        distances = distance.cdist([centroid], labels_embeddings[label], "cosine")[0]
        distances = [e for e in distances if e < 1]
        sort = [i[0] for i in sorted(enumerate(distances), key=lambda x:x[1])]
        print("==============")
        print(SPEECH_CLASSES[int(label)])
        for i in range(10):
            no_links = remove_links(label_posts[label][sort[i]])
            no_ats = remove_ats(no_links)
            remove_repeating = remove_consecutive_phrases(no_ats.split())
            remove_repeating = " ".join(remove_repeating)
            print(f"{i+1}: {remove_repeating}")
    #min_distance = distances[min_index]
    #max_similarity = 1 - min_distance
    #print(min_distance)

    # tokens = tokenize(data)
    # embedding = model_gn[tokens[0]]
    #
    # print(model_gn.most_similar(tokens[0]))
    # print(model_gn.similar_by_vector(embedding))

