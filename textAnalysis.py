import pickle

import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from nltk.corpus import stopwords

from textProcessing import tokenize_and_stem, vocabulary_frame
from speech_classes import SPEECH_CLASSES


# todo: vrne you\'re v ofsCls namesto you're
def combine_texts(tabs):
    num_classes = len(SPEECH_CLASSES)

    def read_table(table):
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
            speech_classes[id] += text + " "
        with open(filename, "wb") as f:
            pickle.dump(speech_classes, f)
        return speech_classes

    speech_classes = ["" for _ in range(num_classes)]
    for t in tabs:
        speech_classes_t = read_table(t)
        for i in range(len(speech_classes)):
            speech_classes[i] += speech_classes_t[i]

    return speech_classes


def tf_idf(texts):
    filename_vectorizer = f"model_pickles/tfidf_vectorizer.p"
    filename_tfidf = f"model_pickles/tfidf_vector.p"
    try:
        vect = joblib.load(filename_vectorizer)
        tfidf = joblib.load(filename_tfidf)
        print(f"Loaded TfIdf vector from disk")
        return tfidf, vect.get_feature_names()
    except FileNotFoundError:
        vect = TfidfVectorizer(
            max_features=200000,
            stop_words=set(stopwords.words("english")),
            max_df=0.8,
            min_df=5,
            use_idf=True,
            tokenizer=tokenize_and_stem,
            ngram_range=(1, 3)
        )
        print(f"Performing TfIdf...")
        tfidf = vect.fit_transform(texts)
        joblib.dump(vect, filename_vectorizer)
        joblib.dump(tfidf, filename_tfidf)
        return tfidf, vect.get_feature_names()


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
        km = KMeans(n_clusters=k, init="k-means++", max_iter=100, n_init=1)
        km.fit(matrix)
        joblib.dump(km, filename)
        return km


if __name__ == '__main__':
    k = 5
    tables = [f"{i}.csv" for i in [9, 25, 26, 31, 32]]
    ofsCls = combine_texts(tables)
    tfidf, terms = tf_idf(ofsCls)
    dense = tfidf.todense()
    denselist = dense.tolist()

    all_keywords = []
    for description in denselist:
        x = 0
        keywords = []
        for word in description:
            if word > 0:
                keywords.append(terms[x])
            x += 1
        all_keywords.append(keywords)

    km = k_means(tfidf, k)
    labels = [SPEECH_CLASSES[i] for i in km.labels_]
    order_centroids = km.cluster_centers_.argsort()[:, ::-1]

    with open("clusters.txt", "w") as f:
        for i in range(k):
            f.write(f"Cluster {i}\n")
            for ind in order_centroids[i, :10]:
                f.write(f"\t{terms[ind]}\n")
            f.write("\n\n")

    kmean_indices = km.fit_predict(tfidf)
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
    for i in range(len(SPEECH_CLASSES)):
        ax.annotate(SPEECH_CLASSES[i], (x_axis[i], y_axis[i]))

    plt.show()

    print("")


