import fasttext.util
import gensim.downloader
from gensim.models.fasttext import load_facebook_vectors
from sklearn.manifold import TSNE
from sklearn.manifold import MDS
from sklearn.decomposition import PCA
import numpy as np
import joblib
import os
from statistics import mean
from text_analysis import combine_texts, get_keywords
from speech_classes import SPEECH_CLASSES

import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sentence_transformers.util import pytorch_cos_sim
from matplotlib.axes._axes import _log as matplotlib_axes_logger
matplotlib_axes_logger.setLevel('ERROR')


def plot_dense_embeddings(title, labels, embedding_clusters, filename=None):
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
    plot_dense_embeddings(title, fixed_labels, embeddings_en_2d, filename)


def plotMDS(title, embedding_clusters, filename=None):
    embedding_clusters = np.array(embedding_clusters)
    n, m, k = embedding_clusters.shape
    print("Performing MDS")
    model_en_2d = MDS(n_components=2, max_iter=3500, random_state=32)
    model_en_2d = model_en_2d.fit_transform(embedding_clusters.reshape(n * m, k))
    embeddings_en_2d = np.array(model_en_2d).reshape(n, m, 2)
    plot_dense_embeddings(title, fixed_labels, embeddings_en_2d, filename)


def plotPCA(title, embedding_clusters, filename=None):
    embedding_clusters = np.array(embedding_clusters)
    n, m, k = embedding_clusters.shape
    print("Performing PCA")
    model_en_2d = PCA(n_components=2, random_state=32)
    model_en_2d = model_en_2d.fit_transform(embedding_clusters.reshape(n * m, k))
    embeddings_en_2d = np.array(model_en_2d).reshape(n, m, 2)
    plot_dense_embeddings(title, fixed_labels, embeddings_en_2d, filename)


keywords_dir = "w2v_term_analysis"

if __name__ == '__main__':

    try:
        label_keywords = joblib.load(os.path.join("w2v_term_analysis", "keywords.p"))
    except FileNotFoundError:
        tables = [f"{i}.csv" for i in [9, 21, 25, 26, 31, 32, 'jigsaw-toxic']]
        documents, classes = combine_texts(tables)
        keywords = get_keywords(documents)
        label_keywords = {SPEECH_CLASSES[classes[i]]: keywords[i] for i in range(len(keywords))}
        if not os.path.exists(keywords_dir):
            os.mkdir(keywords_dir)
        joblib.dump(label_keywords, os.path.join(keywords_dir, "keywords.p"))

    fixed_labels = list(label_keywords.keys())
    manually_ordered = ['sexist', 'appearance-related', 'offensive', 'homophobic',
                        'racist', 'abusive', 'intellectual', 'threat', 'severe_toxic', 'identity_hate',
                        'hateful', 'political', 'religion', 'profane', 'obscene', 'insult',
                        'toxic',  'cyberbullying']
    fixed_labels = [label for label in manually_ordered if label in fixed_labels]
    print(fixed_labels)
    for label, keywords in label_keywords.items():
        for i in range(len(keywords)):
            keywords[i] = (max(keywords[i], key=lambda x: keywords[i][x]))

    if not os.path.exists('cc.en.300.bin'):
        fasttext.util.download_model('en', if_exists='ignore')  # English
        ft = fasttext.load_model('cc.en.300.bin')

    models = [
        lambda: gensim.downloader.load('word2vec-google-news-300'),
        lambda: gensim.downloader.load('glove-wiki-gigaword-300'),
        lambda: load_facebook_vectors("cc.en.300.bin")
    ]

    for model in models:
        print("Loading model...")
        model_gn = model()
        print("Loaded")
        embedding_clusters = list()
        for key in fixed_labels:
            embeddings = [model_gn[word] for word in (label_keywords[key]) if word in model_gn]
            embedding_clusters.append(embeddings)
        min_len = len(min(embedding_clusters, key=len))
        print(min_len)
        embedding_clusters = [cluster[:min_len] for cluster in embedding_clusters]

        embedding_totals = [sum(embeddings)/len(embeddings) for embeddings in embedding_clusters]
        similarity = pytorch_cos_sim(embedding_totals, embedding_totals).numpy()
        np.fill_diagonal(similarity, np.nan)

        embedding_totals = [[tot] for tot in embedding_totals]
        plotPCA("PCA Top Terms", embedding_totals)
        plotMDS("MDS Top Terms", embedding_totals)
        plotTSNE("TSNE Top Terms", embedding_totals)

        plt.pcolor(similarity, cmap='plasma')
        plt.xticks([x + 0.5 for x in range(len(fixed_labels))], fixed_labels, rotation=90)#, ha="right")
        plt.yticks([y + 0.5 for y in range(len(fixed_labels))], fixed_labels)
        plt.colorbar(label="Cosine Similarity", orientation="vertical")
        plt.tight_layout()
        plt.show()













