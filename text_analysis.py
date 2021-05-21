import os
import pickle

import joblib
import pandas as pd
from nltk.corpus import stopwords
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer

from speech_classes import SPEECH_CLASSES
from text_processing import tokenize_and_stem_map_terms, tokenize_and_stem_stopwords, remove_links, remove_ats, \
    remove_retweets, remove_consecutive_phrases


def combine_texts(tabs):
    """
    Returns 2 values. The first is a list of strings, each string representing
    a document of all the data in each speech class. The second is a list of speech class
    id's, in the same order as their respective documents from the first list.
    """

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
    """
    Returns 3 values. The first is the tfidf vectorizer, the second is
    the vectorizer's vocabulary and the third is a dictionary which maps
    stemmed terms to unstemmed terms.
    """

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
            max_df=0.8,
            min_df=0.1,
            use_idf=True,
            tokenizer=tokenizer_function,
            ngram_range=(1, 3)
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
    if not os.path.exists("model_pickles"):
        os.makedirs("model_pickles")
    filename_vectorizer = f"model_pickles/keyword_vectorizer.p"
    filename_tfidf = f"model_pickles/keyword_vector.p"
    filename_stem_term_map = f"model_pickles/keyword_stem_term_map.p"
    try:
        vect = joblib.load(filename_vectorizer)
        tfidf = joblib.load(filename_tfidf)
        stem_term_map = joblib.load(filename_stem_term_map)
        print(f"Loaded keywords TfIdf vector from disk")
    except FileNotFoundError:
        stem_term_map = dict()
        tokenizer_function = lambda t: tokenize_and_stem_map_terms(t, stem_term_map)
        stop_words = tokenize_and_stem_stopwords(set(stopwords.words("english")))
        vect = TfidfVectorizer(
            stop_words=stop_words,
            max_df=0.9,
            min_df=0.1,
            use_idf=True,
            tokenizer=tokenizer_function,
        )
        print(f"Calculating keywords...")
        tfidf = vect.fit_transform(data)
        # remove function to prevent crash, can't pickle lambdas
        vect.tokenizer = None
        joblib.dump(vect, filename_vectorizer)
        joblib.dump(tfidf, filename_tfidf)
        joblib.dump(stem_term_map, filename_stem_term_map)
    finally:
        terms = vect.get_feature_names()
        dense = tfidf.todense()
        denselist = dense.tolist()

        all_keywords = []
        for description in denselist:
            x = 0
            keywords = []
            for frequency in description:
                if frequency > 0:
                    keywords.append((terms[x], frequency))
                x += 1
            keywords = list(sorted(keywords, key=lambda item: item[1], reverse=True))
            all_keywords.append(keywords)

        all_full_terms = list()
        for keywords in all_keywords:
            label_full_terms = list()
            for keyword in keywords:
                full_term = stem_term_map[keyword[0]]
                label_full_terms.append(full_term)
            all_full_terms.append(label_full_terms)

        return all_full_terms
