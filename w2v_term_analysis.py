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


keywords_dir = "w2v_term_analysis"
if not os.path.exists(keywords_dir):
    os.mkdir(keywords_dir)

if __name__ == '__main__':
    try:
        label_keywords = joblib.load(os.path.join("w2v_term_analysis", "keywords.p"))
    except FileNotFoundError:
        tables = [f"{i}.csv" for i in [9, 21, 25, 26, 31, 32, 'jigsaw-toxic']]
        documents, classes = combine_texts(tables)
        keywords = get_keywords(documents)
        label_keywords = {SPEECH_CLASSES[classes[i]]: keywords[i] for i in range(len(keywords))}
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
        fasttext.util.download_model('en', if_exists='ignore')
        ft = fasttext.load_model('cc.en.300.bin')

    models = {
        'Word2vec': lambda: gensim.downloader.load('word2vec-google-news-300'),
        'Glove': lambda: gensim.downloader.load('glove-wiki-gigaword-300'),
        'FastText': lambda: load_facebook_vectors("cc.en.300.bin")
    }

    for model_name, model in models.items():
        print(f"Loading {model_name} model...")
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
        plotPCA("PCA Top Terms", fixed_labels, embedding_totals)
        plotMDS("MDS Top Terms", fixed_labels, embedding_totals)
        plotTSNE("TSNE Top Terms", fixed_labels, embedding_totals)

        plotDistanceMatrix(f"Top Terms Similarity {model_name} embedding", fixed_labels, similarity)













