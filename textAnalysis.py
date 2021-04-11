from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

numberOfClasses = 21

# todo: vrne you\'re v ofsCls namesto you're
def combineTexts(tabs):
    ofsCls = ["" for _ in range(numberOfClasses)]
    for t in tabs:
        print("reading table", t, "...")
        tab = pd.read_csv("data/" + t)
        for row in tab.iterrows():
            id = row[1]["class"]
            text = row[1]["text"]
            ofsCls[id] += text + " "
    return ofsCls


def TfIdf(texts):
    print("calculating Tf-Idf ...")
    vect = TfidfVectorizer()    # parameters for tokenization, stopwords can be passed
    tfidf = vect.fit_transform(texts)

    cosine = (tfidf * tfidf.T).A
    return cosine


tables = ["9.csv", "21.csv", "25.csv"]
ofsCls = combineTexts(tables)
cosine = TfIdf(ofsCls)

print("Cosine similarity between the documents: \n{}".format(cosine))

