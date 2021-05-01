import re

import nltk
import pandas as pd
from nltk.stem import SnowballStemmer
import string

stemmer = SnowballStemmer("english")


def tokenize(raw):
    text = raw.lower()
    # remove punctuation
    table = text.maketrans({key: None for key in string.punctuation})
    text = text.translate(table)
    tokens = nltk.word_tokenize(text)
    return tokens


def stemming(tokens):
    stems = [(token, stemmer.stem(token)) for token in tokens]
    return stems


def tokenize_and_stem(text):
    # Remove links
    url_link_regex = r'https?:\/\/[^\s]*'
    text = re.sub(url_link_regex, '', text)
    # First tokenize by sentence, then by word to ensure that punctuation is caught as it's own token.
    tokens = [word for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []
    # Filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation).
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)
    stems = [stemmer.stem(t) for t in filtered_tokens]
    return stems


# same as above, but saves a mapping from stems to dictionaries of original terms and their counts
# used for retrieving original terms
def tokenize_and_stem_map_terms(text, stem_term_map):
    # Remove links
    url_link_regex = r'https?:\/\/[^\s]*'
    text = re.sub(url_link_regex, '', text)
    # First tokenize by sentence, then by word to ensure that punctuation is caught as it's own token.
    tokens = [word for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []
    # Filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation).
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)

    stems = list()
    for t in filtered_tokens:
        stem = stemmer.stem(t)
        stems.append(stem)
        if stem not in stem_term_map:
            stem_term_map[stem] = dict()
        stem_term_map[stem][t] = stem_term_map[stem].get(t, 0) + 1
    return stems


def vocabulary_frame(documents):
    totalvocab_stemmed = []
    totalvocab_tokenized = []
    for i in documents:
        allwords_stemmed = tokenize_and_stem(i)
        totalvocab_stemmed.extend(allwords_stemmed)

        allwords_tokenized = tokenize(i)
        totalvocab_tokenized.extend(allwords_tokenized)

    # Todo: fix length mismatch
    vocab_frame = pd.DataFrame({'words': totalvocab_tokenized}, index=totalvocab_tokenized)
    return vocab_frame

### use case
# text = "John works at OBI."
# tokens = tokenization(text)
# stems = stemming(tokens)
# print(stems)


