import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from sklearn.manifold import TSNE, MDS
from sklearn.decomposition import PCA
from statistics import mean
from matplotlib.axes._axes import _log as matplotlib_axes_logger
matplotlib_axes_logger.setLevel('ERROR')


def plot_dense_embeddings(title, labels, embedding_clusters, filename=None):
    plt.figure(figsize=(16, 9))
    colors = cm.rainbow(np.linspace(0, 1, len(labels)))
    for label, embeddings, color in zip(labels, embedding_clusters, colors):
        x = embeddings[:, 0]
        y = embeddings[:, 1]
        plt.scatter(x, y, c=color, alpha=0.7, label=label, s=200)
        plt.annotate(label.upper(), alpha=1.0, xy=(mean(x), mean(y)), xytext=(0, 0),
                     textcoords='offset points', ha='center', va='center', size=15)
    # plt.legend(loc=4)
    plt.title(title)
    plt.grid(False)
    if filename:
        plt.savefig(f"{filename}.png", format='png', dpi=150, bbox_inches='tight')
    plt.show()


def plotTSNE(title, labels, embedding_clusters, perplexity=15, filename=None):
    embedding_clusters = np.array(embedding_clusters)
    n, m, k = embedding_clusters.shape
    print("Performing TSNE")
    model_en_2d = TSNE(perplexity=perplexity, n_components=2, init='pca', n_iter=3500, random_state=32)
    model_en_2d = model_en_2d.fit_transform(embedding_clusters.reshape(n * m, k))
    embeddings_en_2d = np.array(model_en_2d).reshape(n, m, 2)
    plot_dense_embeddings(title, labels, embeddings_en_2d, filename)


def plotMDS(title, labels, embedding_clusters, filename=None):
    embedding_clusters = np.array(embedding_clusters)
    n, m, k = embedding_clusters.shape
    print("Performing MDS")
    model_en_2d = MDS(n_components=2, max_iter=3500, random_state=32)
    model_en_2d = model_en_2d.fit_transform(embedding_clusters.reshape(n * m, k))
    embeddings_en_2d = np.array(model_en_2d).reshape(n, m, 2)
    plot_dense_embeddings(title, labels, embeddings_en_2d, filename)


def plotPCA(title, labels, embedding_clusters, filename=None):
    embedding_clusters = np.array(embedding_clusters)
    n, m, k = embedding_clusters.shape
    print("Performing PCA")
    model_en_2d = PCA(n_components=2, random_state=32)
    model_en_2d = model_en_2d.fit_transform(embedding_clusters.reshape(n * m, k))
    embeddings_en_2d = np.array(model_en_2d).reshape(n, m, 2)
    plot_dense_embeddings(title, labels, embeddings_en_2d, filename)


def plotDistanceMatrix(title, labels, similarity_matrix, filename=None):
    plt.figure(figsize=(9, 9))
    plt.title(title)
    plt.pcolor(similarity_matrix, cmap='plasma')
    plt.xticks([x + 0.5 for x in range(len(labels))], labels, rotation=90)
    plt.yticks([y + 0.5 for y in range(len(labels))], labels)
    plt.colorbar(label="Cosine Similarity", orientation="vertical")
    plt.tight_layout()
    if filename:
        plt.savefig(f"{filename}.png", format='png', dpi=150, bbox_inches='tight')
    plt.show()
