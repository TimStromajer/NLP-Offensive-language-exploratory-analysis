import fasttext.util
import gensim.downloader
import numpy as np
import joblib
import os
from gensim.models.fasttext import load_facebook_vectors
from sentence_transformers.util import pytorch_cos_sim

from speech_classes import SPEECH_CLASSES
from dense_plotting import plotPCA, plotMDS, plotTSNE, plotDistanceMatrix
from text_analysis import combine_texts, get_keywords
from w2v_document_embeddings import load_or_create


keywords_dir = "w2v_term_analysis"
if not os.path.exists(keywords_dir):
    os.mkdir(keywords_dir)


def keywords_from_tfidf(tables):
    documents, classes = combine_texts(tables)
    keywords = get_keywords(documents)
    label_keywords = {SPEECH_CLASSES[classes[i]]: keywords[i] for i in range(len(keywords))}
    return label_keywords


def create_embedding_clusters(model, label_keywords):
    print(f"Loading model...")
    model_gn = model()
    print("Loaded")
    embedding_clusters = dict()
    for label, keywords in label_keywords.items():
        embeddings = [model_gn[word] for word in (keywords) if word in model_gn]
        embedding_clusters[label] = embeddings
    return embedding_clusters


if __name__ == '__main__':
    tables = [f"{i}.csv" for i in [9, 21, 25, 26, 31, 32, 'jigsaw-toxic']]
    label_keywords = load_or_create(os.path.join(keywords_dir, "keywords.p"),
                                    lambda: keywords_from_tfidf(tables))

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

    def load_fasttext():
        if not os.path.exists('cc.en.300.bin'):
            fasttext.util.download_model('en', if_exists='ignore')
            ft = fasttext.load_model('cc.en.300.bin')
        return load_facebook_vectors("cc.en.300.bin")

    models = {
        'Word2vec': lambda: gensim.downloader.load('word2vec-google-news-300'),
        'Glove': lambda: gensim.downloader.load('glove-wiki-gigaword-300'),
        'FastText': lambda: load_fasttext()
    }

    for model_name, model in models.items():
        print(model_name)

        embedding_clusters = load_or_create(os.path.join(keywords_dir, f"{model_name} keyword embeddings.p"),
                                            lambda: create_embedding_clusters(model, label_keywords))

        embedding_clusters = [embedding_clusters[label] for label in fixed_labels]
        min_len = len(min(embedding_clusters, key=len))
        print(min_len)
        embedding_clusters = [cluster[:min_len] for cluster in embedding_clusters]

        embedding_totals = [sum(embeddings)/len(embeddings) for embeddings in embedding_clusters]
        similarity = pytorch_cos_sim(embedding_totals, embedding_totals).numpy()
        np.fill_diagonal(similarity, np.nan)

        embedding_totals = [[tot] for tot in embedding_totals]
        plotPCA(f"PCA Top Terms {model_name} embedding", fixed_labels, embedding_totals,
                filename=os.path.join(keywords_dir, f"{model_name} PCA"))
        plotMDS(f"MDS Top Terms {model_name} embedding", fixed_labels, embedding_totals,
                filename=os.path.join(keywords_dir, f"{model_name} MDS"))
        plotTSNE(f"TSNE Top Terms {model_name} embedding", fixed_labels, embedding_totals,
                 filename=os.path.join(keywords_dir, f"{model_name} TSNE"))

        plotDistanceMatrix(f"Top Terms Similarity {model_name} embedding", fixed_labels, similarity,
                           filename=os.path.join(keywords_dir, f"{model_name} similarity"))
