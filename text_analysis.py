import os
import pickle
from collections import defaultdict

import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from nltk.corpus import stopwords

from text_processing import tokenize_and_stem, vocabulary_frame, tokenize_and_stem_map_terms, remove_ats, remove_links, \
    remove_consecutive_phrases, tokenize_and_stem_stopwords, remove_retweets, to_ascii
from speech_classes import SPEECH_CLASSES


# todo: vrne you\'re v ofsCls namesto you're
def combine_texts(tabs):
    num_classes = len(SPEECH_CLASSES)

    def read_table(table):
        if not os.path.exists("data_pickles"):
            os.makedirs("data_pickles")
        filename = f"data_pickles/{table}.p"
        try:
            with open(filename, "rb") as f:
                speech_classes = pickle.load(f)
                print("Read", table, "from pickle.")
                return speech_classes
        except FileNotFoundError:
            pass

        speech_classes = ["" for _ in range(num_classes)]
        print(f"No pickle found for {table}, reading from CSV.")

        table_data = pd.read_csv("data/" + table)
        for row in table_data.iterrows():
            id = row[1]["class"]
            text = row[1]["text"]
            text = remove_links(text)
            text = remove_ats(text)
            text = remove_retweets(text)
            text = remove_consecutive_phrases(text.split())
            text = " ".join(text)
            speech_classes[id] += text + " "
        with open(filename, "wb") as f:
            pickle.dump(speech_classes, f)
        return speech_classes

    speech_class_documents = ["" for _ in range(num_classes)]
    non_empty_classes = list(range(len(SPEECH_CLASSES)))
    non_empty_documents = []
    for t in tabs:
        speech_classes_t = read_table(t)
        for i in range(len(speech_class_documents)):
            speech_class_documents[i] += speech_classes_t[i]

    for i in range(len(speech_class_documents)):
        if len(speech_class_documents[i]) == 0:
            non_empty_classes.remove(i)
        else:
            non_empty_documents.append(speech_class_documents[i])

    return non_empty_documents, non_empty_classes


def tf_idf(texts):
    if not os.path.exists("model_pickles"):
        os.makedirs("model_pickles")
    filename_vectorizer = f"model_pickles/tfidf_vectorizer.p"
    filename_tfidf = f"model_pickles/tfidf_vector.p"
    filename_stem_term_map = f"model_pickles/stem_term_map.p"
    try:
        vect = joblib.load(filename_vectorizer)
        tfidf = joblib.load(filename_tfidf)
        stem_term_map = joblib.load(filename_stem_term_map)
        print(f"Loaded TfIdf vector from disk")
    except FileNotFoundError:
        stem_term_map = dict()
        tokenizer_function = lambda t: tokenize_and_stem_map_terms(t, stem_term_map)
        stop_words = tokenize_and_stem_stopwords(set(stopwords.words("english")))
        vect = TfidfVectorizer(
            stop_words=stop_words,
            max_df=0.5,
            min_df=5,
            use_idf=True,
            tokenizer=tokenizer_function,
            ngram_range=(1, 1)
        )
        print(f"Performing TfIdf...")
        tfidf = vect.fit_transform(texts)
        # remove function to prevent crash, can't pickle lambdas
        vect.tokenizer = None
        joblib.dump(vect, filename_vectorizer)
        joblib.dump(tfidf, filename_tfidf)
        joblib.dump(stem_term_map, filename_stem_term_map)
    finally:
        return tfidf, vect.get_feature_names(), stem_term_map


def cosine_distance(tfidf):
    return (tfidf * tfidf.T).A


def k_means(matrix, k):
    filename = f"model_pickles/k{k}.p"
    try:
        km = joblib.load(filename)
        print(f"Loaded K-Means cluster with k={k} from disk")
        return km
    except FileNotFoundError:
        print(f"Performing K-Means clustering with k={k}...")
        km = KMeans(n_clusters=k, init="k-means++", max_iter=500, n_init=500)
        km.fit(matrix)
        joblib.dump(km, filename)
        return km


def get_keywords(data):
    filename = f"model_pickles/keywords.p"
    if not os.path.exists("model_pickles"):
        os.makedirs("model_pickles")
    try:
        keywords = joblib.load(filename)
        print(f"Loaded keywords from disk")
    except FileNotFoundError:
        print("Calculating keywords...")
        docs = defaultdict(list)
        for table in data:
            table_data = pd.read_csv("data/" + table)
            for row in table_data.iterrows():
                id = row[1]["class"]
                text = row[1]["text"]
                text = remove_links(text)
                text = remove_ats(text)
                text = remove_retweets(text)
                text = remove_consecutive_phrases(text.split())
                text = " ".join(text)
                docs[id].append(text)

        stop_words = tokenize_and_stem_stopwords(set(stopwords.words("english")))
        keywords = {key: dict() for key in docs.keys()}
        for speech_class in docs.keys():
            cv = CountVectorizer(
                stop_words=stop_words,
                max_df=0.85,
                tokenizer=tokenize_and_stem,
            )
            cv.fit_transform(docs[speech_class])
            vocab = cv.vocabulary_
            for word in vocab.keys():
                if word in keywords[speech_class]:
                    keywords[speech_class][word] += vocab[word]
                else:
                    keywords[speech_class][word] = vocab[word]

        for speech_class in docs.keys():
            keywords[speech_class] = {k: v for k, v in sorted(keywords[speech_class].items(), key=lambda item: -item[1])}
        joblib.dump(keywords, filename)
    finally:
        return keywords


def main():
    k = 6
    tables = [f"{i}.csv" for i in [9, 21, 25, 26, 31, 32, 'jigsaw-toxic']]
    keywords = get_keywords(tables)

    documents, classes = combine_texts(tables)
    tfidf, terms, stem_term_map = tf_idf(documents)

    km = k_means(tfidf, k)
    kmean_indices = km.fit_predict(tfidf)
    order_centroids = km.cluster_centers_.argsort()[:, ::-1]

    # Creates dict mapping cluster to text classes
    # kmean_indices: index = text class; value = cluster
    cluster_classes = dict()
    for text_class, cluster in enumerate(kmean_indices):
        if cluster not in cluster_classes:
            cluster_classes[cluster] = list()
        cluster_classes[cluster].append(SPEECH_CLASSES[classes[text_class]])

    with open("clusters.txt", "w") as f:
        for i in range(k):
            f.write(f"Cluster {i}: {', '.join(cluster_classes[i])}\n")
            cluster_terms = list()
            for ind in order_centroids[i, :20]:
                output_term = [stem_term_map[term] for term in terms[ind].split()]
                output_term = [max(term, key=term.get) for term in output_term]
                cluster_terms.extend(output_term)
            f.write(f"\t{cluster_terms}\n")
            f.write("\n\n")

    pca = PCA(n_components=2)
    scatter_plot_points = pca.fit_transform(tfidf.toarray())
    colors = ["r", "g", "b", "c", "y", "m"]

    x_axis = []
    y_axis = []
    for x, y in scatter_plot_points:
        x_axis.append(x)
        y_axis.append(y)

    fig, ax = plt.subplots()
    ax.scatter(x_axis, y_axis, c=[colors[d] for d in kmean_indices])
    for i in range(len(classes)):
        ax.annotate(SPEECH_CLASSES[classes[i]], (x_axis[i], y_axis[i]))

    plt.show()

    print("")


if __name__ == '__main__':
    main()
