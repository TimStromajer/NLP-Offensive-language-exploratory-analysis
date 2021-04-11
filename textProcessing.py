import nltk
from nltk.stem import SnowballStemmer

import string


def tokenization(raw):
    text = raw.lower()
    # remove punctuation
    table = text.maketrans({key: None for key in string.punctuation})
    text = text.translate(table)
    tokens = nltk.word_tokenize(text)
    return tokens

def stemming(tokens):
    stemmer = SnowballStemmer("english")
    stems = [(token, stemmer.stem(token)) for token in tokens]
    return stems

### use case
# text = "John works at OBI."
# tokens = tokenization(text)
# stems = stemming(tokens)
# print(stems)


