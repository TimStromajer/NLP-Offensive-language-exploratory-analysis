import fasttext.util
import gensim.downloader
import numpy as np
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


# Create embeddings for keywords for all labels
# model: model that can be indexed with a string and returns the dense embedding
# label keywords:
def create_embedding_clusters(model, label_keywords):
    print(f"Loading model...")
    model_gn = model()
    print("Loaded")
    embedding_clusters = dict()
    top_keywords = dict()
    for label, keywords in label_keywords.items():
        embeddings = [model_gn[word] for word in keywords if word in model_gn]
        embedding_clusters[label] = embeddings
        label_words = [word for word in keywords if word in model_gn]
        top_keywords[label] = label_words
    return embedding_clusters, top_keywords


if __name__ == '__main__':
    tables = [f"{i}.csv" for i in [9, 21, 25, 26, 31, 32, 'jigsaw-toxic']]
    # Get keywords from files, returns dictionary {label: [dictionary{keyword_form: count}, ]}
    label_keywords = load_or_create(os.path.join(keywords_dir, "keywords.p"),
                                    lambda: keywords_from_tfidf(tables))

    fixed_labels = list(label_keywords.keys())
    # Manual ordering for a clearer visualization of similarities
    manually_ordered = ['sexist', 'appearance-related', 'offensive', 'homophobic',
                        'racist', 'abusive', 'intellectual', 'threat', 'severe_toxic', 'identity_hate',
                        'hateful', 'political', 'religion', 'profane', 'obscene', 'insult',
                        'toxic',  'cyberbullying']
    fixed_labels = [label for label in manually_ordered if label in fixed_labels]

    # Get only the keyword form that appears the most often
    for label, keywords in label_keywords.items():
        for i in range(len(keywords)):
            keywords[i] = (max(keywords[i], key=lambda x: keywords[i][x]))

    # Logic for loading models
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

        embedding_clusters, top_keywords = load_or_create(os.path.join(keywords_dir, f"{model_name} keyword embeddings.p"),
                                            lambda: create_embedding_clusters(model, label_keywords))
        # Convert to list of lists
        embedding_clusters = [embedding_clusters[label] for label in fixed_labels]
        top_keywords = [top_keywords[label] for label in fixed_labels]
        # Only include the number of keywords from the smallest class
        # min_len = len(min(embedding_clusters, key=len))
        min_len = 50
        print(min_len)
        embedding_clusters = [cluster[:min_len] for cluster in embedding_clusters]

        for i, keywords in enumerate(top_keywords):
            print(fixed_labels[i])
            print(keywords[:min_len])

        # Calculate the combined embedding for each class
        embedding_totals = [sum(embeddings)/len(embeddings) for embeddings in embedding_clusters]
        similarity = pytorch_cos_sim(embedding_totals, embedding_totals).numpy()
        np.fill_diagonal(similarity, np.nan)

        # Plot
        embedding_totals = [[tot] for tot in embedding_totals]
        plotPCA(f"PCA Top Terms {model_name} embedding", fixed_labels, embedding_totals,
                filename=os.path.join(keywords_dir, f"{model_name} PCA"))
        plotMDS(f"MDS Top Terms {model_name} embedding", fixed_labels, embedding_totals,
                filename=os.path.join(keywords_dir, f"{model_name} MDS"))
        #plotTSNE(f"TSNE Top Terms {model_name} embedding", fixed_labels, embedding_totals,
        #         filename=os.path.join(keywords_dir, f"{model_name} TSNE"))

        plotDistanceMatrix(f"Top Terms Similarity {model_name} embedding", fixed_labels, similarity,
                           filename=os.path.join(keywords_dir, f"{model_name} similarity"))
