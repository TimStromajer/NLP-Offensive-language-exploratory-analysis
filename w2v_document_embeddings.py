import gensim.downloader
import gensim
import pandas as pd
import os
import pickle
import numpy as np
import torch

from lazy_load import lazy
from scipy.spatial import distance
from text_processing import tokenize, remove_links, remove_ats, remove_consecutive_phrases
from speech_classes import SPEECH_CLASSES
from nltk.corpus import stopwords


from sklearn.manifold import TSNE
from sklearn.manifold import MDS
from sklearn.decomposition import PCA
from statistics import mean
pd.options.display.max_columns = None
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sentence_transformers.util import pytorch_cos_sim

stopwords = stopwords.words('english')


def read_document(file_path):
    return pd.read_csv(file_path, dtype=str, usecols=[1, 2])


def embed_document(model, tokens):
    total_vector = np.zeros([model.vector_size], dtype=np.float32)
    normalization = 0
    for token in tokens:
        if token in stopwords:
            # print(f"skipped: {token}")
            continue
        if token in model:
            embedding = model[token]
            total_vector += embedding
            normalization += 1
    if normalization == 0:
        return total_vector
    return total_vector / normalization


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


def document_group_similarities(a_embeddings, b_embeddings):
    distances = pytorch_cos_sim(a_embeddings, b_embeddings)
    a_to_b_averages = torch.mean(distances, dim=1)
    # print("Correct dimension:", len(a_embeddings) == len(a_to_b_averages))
    b_to_a_averages = torch.mean(distances, dim=0)
    top_a_to_b = torch.argsort(a_to_b_averages, descending=True)
    top_b_to_a = torch.argsort(b_to_a_averages, descending=True)
    # print([a_to_b_averages[i.item()] for i in top_a_to_b[:5]])
    # print([b_to_a_averages[i.item()] for i in top_b_to_a[:5]])
    a_to_b = torch.mean(a_to_b_averages)
    b_to_a = torch.mean(b_to_a_averages)
    return a_to_b, b_to_a, top_a_to_b, top_b_to_a


def plot_similar_words(title, labels, embedding_clusters, filename=None):
    plt.figure(figsize=(16, 9))
    colors = cm.rainbow(np.linspace(0, 1, len(labels)))
    for label, embeddings, color in zip(labels, embedding_clusters, colors):
        x = embeddings[:, 0]
        y = embeddings[:, 1]
        plt.scatter(x, y, c=color, alpha=0.7, label=label)
        plt.annotate(label.upper(), alpha=1.0, xy=(mean(x), mean(y)), xytext=(0, 0),
                     textcoords='offset points', ha='center', va='center', size=15)
    plt.legend(loc=4)
    plt.title(title)
    plt.grid(False)
    if filename:
        plt.savefig(filename, format='png', dpi=150, bbox_inches='tight')
    plt.show()


def plotTSNE(title, embedding_clusters, perplexity=15, filename=None):
    embedding_clusters = np.array(embedding_clusters)
    n, m, k = embedding_clusters.shape
    print("Performing TSNE")
    model_en_2d = TSNE(perplexity=perplexity, n_components=2, init='pca', n_iter=3500, random_state=32)
    model_en_2d = model_en_2d.fit_transform(embedding_clusters.reshape(n * m, k))
    embeddings_en_2d = np.array(model_en_2d).reshape(n, m, 2)
    plot_similar_words(title, [SPEECH_CLASSES[int(label)] for label in labels], embeddings_en_2d, filename)



def plotMDS(title, embedding_clusters, filename = None):
    embedding_clusters = np.array(embedding_clusters)
    n, m, k = embedding_clusters.shape
    print("Performing MDS")
    model_en_2d = MDS(n_components=2, max_iter=3500, random_state=32)
    model_en_2d = model_en_2d.fit_transform(embedding_clusters.reshape(n * m, k))
    embeddings_en_2d = np.array(model_en_2d).reshape(n, m, 2)
    plot_similar_words(title, [SPEECH_CLASSES[int(label)] for label in labels], embeddings_en_2d, filename)


def plotPCA(title, embedding_clusters, filename = None):
    embedding_clusters = np.array(embedding_clusters)
    n, m, k = embedding_clusters.shape
    model_en_2d = PCA(n_components=2, random_state = 32)
    model_en_2d = model_en_2d.fit_transform(embedding_clusters.reshape(n * m, k))
    embeddings_en_2d = np.array(model_en_2d).reshape(n, m, 2)
    plot_similar_words(title, [SPEECH_CLASSES[int(label)] for label in labels], embeddings_en_2d, filename)


intermediate_location = f"{os.path.basename(__file__)}-intermediate_data"
if not os.path.exists(intermediate_location):
    os.mkdir(intermediate_location)


if __name__ == '__main__':
    regenerate = False
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
        data = data[data['class'] != '0']
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

    labels = list(labels_embeddings.keys())
    print([SPEECH_CLASSES[int(label)] for label in labels])

    top_count_print = 5
    top_count_visualize = 30

    # Centroid based representative
    embedding_totals = list()
    embedding_top_similar = list()
    posts_top_similar = list()

    for label in labels_embeddings:
        combined = sum(labels_embeddings[label])
        combined = combined/len(labels_embeddings[label])
        embedding_totals.append(combined)
        count = len(labels_embeddings[label])
        centroid = combined/count
        distances = distance.cdist([centroid], labels_embeddings[label], "cosine")[0]
        distances = [e for e in distances if e < 1]
        sort = [i[0] for i in sorted(enumerate(distances), key=lambda x:x[1])]
        print("==============")
        print(SPEECH_CLASSES[int(label)])
        for i in range(top_count_print):
            no_links = remove_links(label_posts[label][sort[i]])
            no_ats = remove_ats(no_links)
            remove_repeating = remove_consecutive_phrases(no_ats.split())
            remove_repeating = " ".join(remove_repeating)
            print(f"{i+1}: {remove_repeating}")

        label_embedding_top = list()
        label_post_top = list()
        for i in range(top_count_visualize):
            label_embedding_top.append(labels_embeddings[label][sort[i]])
            label_post_top.append(label_posts[label][sort[i]])
        embedding_top_similar.append(label_embedding_top)
        posts_top_similar.append(label_post_top)

    #Centroid based similarity
    similarity = pytorch_cos_sim(embedding_totals, embedding_totals)
    similarity = similarity.numpy()
    print([SPEECH_CLASSES[int(label)] for label in labels])
    #print(similarity)



    plotMDS("Totals", [[tot] for tot in embedding_totals])
    # plotMDS("Title", embedding_top_similar)
    # plotTSNE("Title", embedding_top_similar, perplexity=10)
    # plotPCA("Title", embedding_top_similar)

    plt.pcolor(similarity)
    plt.xticks([x + 0.5 for x in range(len(labels))], [SPEECH_CLASSES[int(label)] for label in labels], rotation=45, ha="right")
    plt.yticks([y + 0.5 for y in range(len(labels))], [SPEECH_CLASSES[int(label)] for label in labels])
    plt.colorbar(label="Cosine Similarity", orientation="vertical")
    plt.tight_layout()
    plt.show()

    label_distances = np.zeros((len(labels), len(labels)), dtype=np.float32)

    #Distance based representative
    for i, (label, embeddings) in enumerate(labels_embeddings.items()):
        label_distance, _, indexs, _ = document_group_similarities(embeddings, embeddings)
        print("==============")
        print(SPEECH_CLASSES[int(label)])
        print(label_distance)
        label_distances[i, i] = label_distance
        for j in range(top_count_print):
            no_links = remove_links(label_posts[label][indexs[j].item()])
            no_ats = remove_ats(no_links)
            remove_repeating = remove_consecutive_phrases(no_ats.split())
            remove_repeating = " ".join(remove_repeating)
            print(f"{j+1}: {remove_repeating}")
