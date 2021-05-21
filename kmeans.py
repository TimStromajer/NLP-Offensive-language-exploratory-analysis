import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from dense_plotting import format_label
from speech_classes import SPEECH_CLASSES
from text_analysis import combine_texts, tf_idf, k_means


def main():
    k = 6
    tables = [f"{i}.csv" for i in [9, 21, 25, 26, 31, 32, 'jigsaw-toxic']]
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
            for ind in order_centroids[i, :10]:
                output_term = [stem_term_map[term] for term in terms[ind].split()]
                output_term = [max(term, key=term.get) for term in output_term]
                output_term = " ".join(output_term)
                cluster_terms.append(output_term)
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

    plt.figure(figsize=(8, 7))
    plt.scatter(x_axis, y_axis, c=[colors[d] for d in kmean_indices], alpha=0.6, s=300)
    for i in range(len(classes)):
        label = format_label(SPEECH_CLASSES[classes[i]])
        annotated = plt.annotate(label, (x_axis[i], y_axis[i]), textcoords='offset points',
                                 ha='center', va='center', size=10)
        annotated.set_path_effects([path_effects.Stroke(linewidth=1.5, foreground='white'),
                                    path_effects.Normal()])
    plt.savefig("kmeans.png", dpi=96)
    plt.show()


if __name__ == '__main__':
    main()
