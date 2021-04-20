import pandas as pd
import sqlite3
from sqlite3 import OperationalError
from sqlalchemy import create_engine
from bs4 import BeautifulSoup, NavigableString, Tag
import json
import os
import re

##### 9
# tab = pd.read_csv("data/9_uo.csv")
# tab2 = tab.replace({'class': {0: 13, 2: 0}})    # 1 -> 1
# tab2 = tab2[["class", "tweet"]]
# tab2 = tab2.rename(columns={'tweet': 'text'})
# tab2.to_csv("data/9.csv")

##### 25
# tab = pd.read_csv("data/25_uo.tsv", sep='\t')
# tab2 = tab.replace({'task_2': {"NONE": 0, "OFFN": 1, "HATE": 13, "PRFN": 7}})
# tab2 = tab2[["task_2", "text"]]
# tab2.to_csv("data/25.csv")

##### 21
# tabData = {'class':  [], 'text': []}
# file = "data/21_uo.json"
# with open(file) as f:
#   data = json.load(f)
#
# for tweet in data:
#     l = data[tweet]["labels"]
#     if len(l) != 3:
#         continue
#     # none
#     if l[0] == 0 and l[1] == 0 and l[2] == 0:
#         tabData["class"].append(0)
#         tabData["text"].append(data[tweet]["tweet_text"])
#     # racist
#     elif (l[0] == 1 and l[1] == 1) or (l[0] == 1 and l[2] == 1) or (l[1] == 1 and l[2] == 1):
#         tabData["class"].append(5)
#         tabData["text"].append(data[tweet]["tweet_text"])
#     # sexist
#     elif (l[0] == 2 and l[1] == 2) or (l[0] == 2 and l[2] == 2) or (l[1] == 2 and l[2] == 2):
#         tabData["class"].append(16)
#         tabData["text"].append(data[tweet]["tweet_text"])
#     # homophobe
#     elif (l[0] == 3 and l[1] == 3) or (l[0] == 3 and l[2] == 3) or (l[1] == 3 and l[2] == 3):
#         tabData["class"].append(6)
#         tabData["text"].append(data[tweet]["tweet_text"])
#     # religion
#     elif (l[0] == 4 and l[1] == 4) or (l[0] == 4 and l[2] == 4) or (l[1] == 4 and l[2] == 4):
#         tabData["class"].append(20)
#         tabData["text"].append(data[tweet]["tweet_text"])
#
# df = pd.DataFrame(tabData, columns=['class', 'text'])
# df.to_csv("data/21.csv")

##### 32
# source_path = os.path.join("data", "32_uo", "Sharing Data")
# class_key = {
#     "appearance": 17,
#     "intelligence": 18,
#     "political": 19,
#     "racial": 5,
#     "sexual": 16
# }
# negative = [float('nan'), 'not sure', 'No', 'NO', 'Not Sure', 'Not sure', 'N', 'no']
# positive = ['Yes', 'YES', 'yes ', 'yes']
# other = ['Other', 'others', 'Others', 'racism']
#
# processed_frames = list()
# for file in os.scandir(source_path):
#     tab = pd.read_csv(file.path)
#     tweet_header = tab.columns[0]
#     decision_header = tab.columns[1]
#     harassment_class = class_key[file.name.lower().split()[0]]
#     tab = tab.replace({decision_header: dict.fromkeys(negative, 0) |
#                                         dict.fromkeys(positive, harassment_class) |
#                                         dict.fromkeys(other, 21)})
#     tab = tab[[decision_header, tweet_header]]
#     tab = tab.rename(columns={tweet_header: 'text', decision_header: 'class'})
#     tab = tab[tab['text'].notna()]
#     processed_frames.append(tab)
# total = pd.concat(processed_frames, ignore_index=True)
# # Multiple data-sets all just classify a single from of harassment
# # They mark 0 even if ti is harassment, but not of the type that dataset covers
# # In order to remove instances which were not tagged as harassment in one set but were in another
# # We find duplicates and remove all non-harassment instances, leaving only the instance from the right dataset
# duplicate_selector = total.duplicated(subset='text', keep=False)
# duplicates = total[duplicate_selector]
# neutral_duplicate_selector = duplicates['class'] == 0
# total = total[~(duplicate_selector & neutral_duplicate_selector)]
# not_negative = total[total['class'] != 0]
# not_negative.to_csv(os.path.join("data", "32.csv"))
# # Will likely have to filter out all 0 as well, because many of them are clearly offensive

##### 31


def extract_message(post):
    soup = BeautifulSoup(post, features="lxml")
    useful_text = list()

    def traverse(element):
        if type(element) is NavigableString:
            text = " ".join((str(element)).split())
            text = re.sub(r'(^|[^\s]+):n|(\b)n(\b\s*)', '', text)
            if len(text.split()) > 2:
                useful_text.append(text)
        else:
            for child in element.children:
                traverse(child)
    traverse(soup)
    if useful_text:
        return max(useful_text, key=lambda x: len(x))
    return ""


source_path = os.path.join("data", "31_uo", "31_uo.csv")
bully_posts = pd.read_csv(source_path, header=None, nrows=632, usecols=[0, 1], names=['topic', 'post'])
posts = pd.read_csv(source_path, header=None, skiprows=632, usecols=[3], names=['text'])
posts['text'] = posts['text'].apply(extract_message)

print(bully_posts)
print(posts)
print(posts['text'][4])