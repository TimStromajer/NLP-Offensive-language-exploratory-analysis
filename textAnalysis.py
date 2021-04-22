import pickle

import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.manifold import MDS

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
    print("calculating Tf-Idf ...")
    vect = TfidfVectorizer(
        max_features=200000,
        stop_words='english',
        use_idf=True,
        tokenizer=tokenize_and_stem,
        ngram_range=(1, 3)
    )

    return vect.fit_transform(texts), vect.get_feature_names()


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
        km = KMeans(n_clusters=k)
        km.fit(matrix)
        joblib.dump(km, filename)
        return km


def getClusterWords(km, vocab_frame, cluster, n=6):
    order_centroids = km.cluster_centers_.argsort()[:, ::-1]
    words = []
    for ind in order_centroids[cluster, :n]:
        words.append(vocab_frame.loc[terms[ind].split(' '), ].values.tolist()[0][0])
    return ", ".join(words)


def visualise_mds(tfidf, km, vocab_frame):
    clusters = km.labels_.tolist()
    dist = cosine_distance(tfidf)
    mds = MDS(n_components=2, dissimilarity="precomputed", random_state=1)

    # Shape of the result will be (n_components, n_samples).
    pos = mds.fit_transform(dist)

    xs, ys = pos[:, 0], pos[:, 1]
    cluster_colors = {0: '#1b9e77', 1: '#d95f02', 2: '#7570b3', 3: '#e7298a', 4: '#66a61e'}

    # Define cluster names
    cluster_names = dict([(i, getClusterWords(km, vocab_frame, i, 3)) for i in range(5)])

    df = pd.DataFrame(dict(x=xs, y=ys, label=clusters, title=SPEECH_CLASSES))

    # Group by cluster.
    groups = df.groupby('label')

    # Set up plot.
    fig, ax = plt.subplots(figsize=(17, 9))  # set size
    ax.margins(0.05)  # Optional, just adds 5% padding to the autoscaling

    # Iterate through groups to layer the plot.
    # Note that we use the cluster_name and cluster_color dicts with the 'name'
    # lookup to return the appropriate color/label.
    for name, group in groups:
        ax.plot(group.x, group.y, marker='o', linestyle='', ms=12,
                label=cluster_names[name], color=cluster_colors[name],
                mec='none')
        ax.set_aspect('auto')
        ax.tick_params(
            axis='x',  # changes apply to the x-axis
            which='both',  # both major and minor ticks are affected
            bottom='off',  # ticks along the bottom edge are off
            top='off',  # ticks along the top edge are off
            labelbottom='off')
        ax.tick_params(
            axis='y',  # changes apply to the y-axis
            which='both',  # both major and minor ticks are affected
            left='off',  # ticks along the bottom edge are off
            top='off',  # ticks along the top edge are off
            labelleft='off')

    ax.legend(numpoints=1)  # show legend with only 1 point

    # Add label in x,y position with the label as the film title.
    for i in range(len(df)):
        ax.text(df.loc[df.index[i], 'x'], df.loc[df.index[i], 'y'], df.loc[df.index[i], 'title'], size=8)

    plt.show()


if __name__ == '__main__':
    tables = [f"{i}.csv" for i in [9, 25, 26, 31]]
    ofsCls = combine_texts(tables)
    vocab_frame = vocabulary_frame(ofsCls)
    tfidf, terms = tf_idf(ofsCls)
    km = k_means(tfidf, 5)
    # visualise_mds(tfidf, km, vocab_frame)

    print("done")

