import gensim.downloader
from sklearn.manifold import TSNE
from sklearn.manifold import MDS
from sklearn.decomposition import PCA
import numpy as np
from statistics import mean
import pandas as pd
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from IPython.display import display
pd.options.display.max_columns = None

import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sentence_transformers.util import pytorch_cos_sim

from matplotlib.axes._axes import _log as matplotlib_axes_logger
matplotlib_axes_logger.setLevel('ERROR')

TOP_N = 30
KEYS = {
    "obscene": ['wikipedia', 'vandalism', 'revert', 'wiki', 'ip', 'homosexual', 'editors', 'unblock', 'billcj', 'wp', 'utc', 'encyclopedia', 'vandalising', 'cking', 'anti-semite', 'wikipedians', 'undo', 'sockpuppet', 'rvv', 'anus'],
    "offensive, abusive, racist": ['hillbilly', 'niggah', 'beaner', 'wrestlemania', 'bird', 'sequel', 'terrifying', 'april', 'rn', 'spic', 'maga', 'hella', 'syria', 'surrender', 'nig', 'retweet', 'nigguh', 'bae', 'booty', 'aye'],
    "profane, hateful": ['cricket', 'impeachtrump', 'boris', 'gloves', 'impeachment', 'dhoni', 'maga', 'congress', 'bengal', 'oral', 'coon', 'tory', 'brexit', 'tournament', 'trump2020', 'notmypresident', 'democracy', 'theresistance', 'mp', 'nig'],
    "political": ['fuckbag', 'dickwad', 'clinton', 'notmypresident', 'ruski', 'cheeto', 'deplorable', 'hrc', 'snowflake', 'cia', 'ned', 'cnn', 'donnie', 'asswipe', 'nomination', 'manufacturing', 'jan', 'ass-hat', 'duchebag', 'sleaze'],
    "cyberbullying": ['riot', 'op', 'forum', 'ip', 'hacker', 'gd', 'it.', 'teammates', 'boost', 'increase', 'silver', 'stacks', 'me.', 'mana', 'summon', 'feedback', 'diamond', 'range', 'dunk', 'queue'],
    "religion": ['muzzie', 'surrender', 'islamist', 'spring', 'maga', 'graham', 'gabrielle', 'jihad', 'syria', 'gaza', 'appease', 'mosque', 'yummy', 'fanatic', 'whoa', 'instant', 'sigh', 'invasion', 'radical', 'raghead']
}
FIXED_KEYS = list(KEYS.keys())

lemmatizer = WordNetLemmatizer()
ps = PorterStemmer()


for key in KEYS.keys():
    lemma = lemmatizer.lemmatize(key)
    stem = ps.stem(key)
    # KEYS[key].add(lemma)
    # KEYS[key].add(stem)
    # KEYS[key].add(key)


def same_word(similar_word, ommit_words):
    similar_word = similar_word.replace("_", " ").replace("-", " ").lower()

    if similar_word in ommit_words:
        print(f"{ommit_words} -- {similar_word}")
        return True

    for ommit_word in ommit_words:
        if ommit_word in similar_word:
            print(f"{ommit_words} -- {similar_word}")
            return True

    return False


def getSimilarWords(model_gn):
    embedding_clusters = []
    word_clusters = []
    for key in FIXED_KEYS:
        ommit_words = KEYS[key]
        embeddings = []
        words = []
        for similar_word, _ in model_gn.most_similar(key, topn=TOP_N * 3):
            if not same_word(similar_word, ommit_words):
                words.append(similar_word)
                embeddings.append(model_gn[similar_word])

        if len(words) < TOP_N or len(embeddings) < TOP_N:
            print("ERROR")

        words = words[:TOP_N]
        embeddings = embeddings[:TOP_N]

        embedding_clusters.append(embeddings)
        word_clusters.append(words)

    return (word_clusters, embedding_clusters)


def displayDF(word_clusters):
    df = pd.DataFrame(dict(zip(FIXED_KEYS, word_clusters)))
    display(df)


def plot_similar_words(title, labels, embedding_clusters, word_clusters, filename=None):
    plt.figure(figsize=(16, 9))
    colors = cm.rainbow(np.linspace(0, 1, len(labels)))
    for label, embeddings, words, color in zip(labels, embedding_clusters, word_clusters, colors):
        x = embeddings[:, 0]
        y = embeddings[:, 1]
        plt.scatter(x, y, c=color, alpha=0.7, label=label)
        for i, word in enumerate(words):
            plt.annotate(word, alpha=0.5, xy=(x[i], y[i]), xytext=(5, 2),
                         textcoords='offset points', ha='right', va='bottom', size=8)
        plt.annotate(label.upper(), alpha=1.0, xy=(mean(x), mean(y)), xytext=(0, 0),
                     textcoords='offset points', ha='center', va='center', size=15)
    plt.legend(loc=4)
    plt.title(title)
    plt.grid(False)
    if filename:
        plt.savefig(filename, format='png', dpi=150, bbox_inches='tight')
    plt.show()


def plotTSNE(title, word_clusters, embedding_clusters, perplexity=15, filename=None):
    embedding_clusters = np.array(embedding_clusters)
    n, m, k = embedding_clusters.shape
    model_en_2d = TSNE(perplexity=perplexity, n_components=2, init='pca', n_iter=3500, random_state=32)
    model_en_2d = model_en_2d.fit_transform(embedding_clusters.reshape(n * m, k))
    embeddings_en_2d = np.array(model_en_2d).reshape(n, m, 2)
    plot_similar_words(title, FIXED_KEYS, embeddings_en_2d, word_clusters, filename)


def plotMDS(title, word_clusters, embedding_clusters, filename = None):
    embedding_clusters = np.array(embedding_clusters)
    n, m, k = embedding_clusters.shape
    model_en_2d = MDS(n_components=2, max_iter=3500, random_state=32)
    model_en_2d = model_en_2d.fit_transform(embedding_clusters.reshape(n * m, k))
    embeddings_en_2d = np.array(model_en_2d).reshape(n, m, 2)
    plot_similar_words(title, FIXED_KEYS, embeddings_en_2d, word_clusters, filename)


def plotPCA(title, word_clusters, embedding_clusters, filename = None):
    embedding_clusters = np.array(embedding_clusters)
    n, m, k = embedding_clusters.shape
    model_en_2d = PCA(n_components=2, random_state = 32)
    model_en_2d = model_en_2d.fit_transform(embedding_clusters.reshape(n * m, k))
    embeddings_en_2d = np.array(model_en_2d).reshape(n, m, 2)
    plot_similar_words(title, FIXED_KEYS, embeddings_en_2d, word_clusters, filename)


if __name__ == '__main__':
    print("Loading model...")
    model_gn = gensim.downloader.load('word2vec-google-news-300')
    print("Loaded")
    word_clusters = list()
    embedding_clusters = list()
    for key in FIXED_KEYS:
        words = [word for word in (KEYS[key])if word in model_gn]
        embeddings = [model_gn[word] for word in words if word in model_gn]
        word_clusters.append(words)
        embedding_clusters.append(embeddings)
    min_len = len(min(embedding_clusters, key=len))
    print(min_len)
    word_clusters = [cluster[:min_len] for cluster in word_clusters]
    embedding_clusters = [cluster[:min_len] for cluster in embedding_clusters]
    # word_clusters, embedding_clusters = getSimilarWords(model_gn)
    # print(np.array(embedding_clusters).shape)

    # plotTSNE("Similar words - Word2Vec [t-SNE]", word_clusters, embedding_clusters, perplexity=5)
    # plotTSNE("Similar words - Word2Vec [t-SNE]", word_clusters, embedding_clusters, perplexity=10)
    # plotTSNE("Similar words - Word2Vec [t-SNE]", word_clusters, embedding_clusters, perplexity=20)
    # plotTSNE("Similar words - Word2Vec [t-SNE]", word_clusters, embedding_clusters, perplexity=40)
    # plotTSNE("Similar words - Word2Vec [t-SNE]", word_clusters, embedding_clusters, perplexity=80)
    # plotMDS("Similar words - Word2Vec [MDS]", word_clusters, embedding_clusters)
    # plotPCA("Similar words - Word2Vec [PCA]", word_clusters, embedding_clusters)

    centroids = [sum(embeddings) for embeddings in embedding_clusters]
    similarity = pytorch_cos_sim(centroids, centroids)
    print(FIXED_KEYS)
    print(similarity)













