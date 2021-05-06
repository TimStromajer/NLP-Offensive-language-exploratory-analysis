import pandas as pd
import sqlite3
from sqlite3 import OperationalError
from sqlalchemy import create_engine
from bs4 import BeautifulSoup, NavigableString, Tag
import json
import tweepy as tw
import os
from speech_classes import SPEECH_CLASSES

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


##### 26 - getting 900 tweets / 15 min = 3600 / hour; last: 848187434735194112
# consumer_key= 'IQ82FYxYl0ujW6ulJROH2GGhk'
# consumer_secret= '5PYcvLKayHKU7QQGtQOW3Rhchf0dT7km5dBMJsRb3CrgPjEKvf'
#
# auth = tw.AppAuthHandler(consumer_key, consumer_secret)
# api = tw.API(auth, wait_on_rate_limit=True)
#
# tabData = {'class':  [], 'text': []}
# tweetCounter = 0
# tab = pd.read_csv("data/26_uo.tab", sep='\t')
# start = False
# for row in tab.iterrows():
#     cls = row[1]["maj_label"]
#     if cls == "abusive":
#
#         idt = int(row[1]["tweet_id"])
#         print(idt)
#
#         if idt == 848187434735194112:
#             start = True
#         if not start:
#             continue
#
#         tweetCounter += 1
#         try:
#             twit = api.get_status(idt, tweet_mode="extended")
#             print("tweet obtained")
#             if len(twit.full_text) > 0:
#                 tabData["class"].append(2)
#                 tabData["text"].append(twit.full_text)
#         except:
#             print("tweet does not exist")
#
#         if tweetCounter > 10000:
#             break
#
#
# df = pd.DataFrame(tabData, columns=['class', 'text'])
# df.to_csv("data/26.csv")
## combine two tables
# tab1 = pd.read_csv("data/26_old.csv")
# tab2 = pd.read_csv("data/26_2.csv")
# for row in tab2.iterrows():
#     tab1.append({"class": row[1]["class"], "text": row[1]["text"]}, ignore_index=True)
#     # tab1["class"].append(row[1]["class"])
#     # tab1["text"].append(row[1]["text"])
# tab1 = tab1[["class", "text"]]
# tab1.to_csv("data/26_combined.csv")


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


##### 31
# def extract_message(post):
#     soup = BeautifulSoup(post, features="lxml")
#     useful_text = list()
#
#     def traverse(element):
#         if type(element) is NavigableString:
#             text = " ".join((str(element)).split())
#             text = re.sub(r'(^|[^\s]+):n|(\b)n(\b\s*)', '', text)
#             if len(text) > 3 and len(text) != 16 and text != "Quote: ":
#                 useful_text.append(text)
#         else:
#             for child in element.children:
#                 traverse(child)
#     traverse(soup)
#
#     return useful_text
#
#
# source_path = os.path.join("data", "31_uo", "31_uo.csv")
# bully_posts = pd.read_csv(source_path, header=None, nrows=632, usecols=[0, 1], names=['topic', 'post'])
# bully_posts['full_id'] = bully_posts['topic'].combine(bully_posts['post'], lambda a, b: f"{a}:{b}")
#
# posts = pd.read_csv(source_path, header=None, skiprows=632, usecols=[0, 1, 3], names=['topic', 'post', 'text'])
# posts['full_id'] = posts['topic'].combine(posts['post'], lambda a, b: f"{a}:{b}")
#
# keep_negative = False
# if keep_negative:
#     posts['class'] = 0
#     posts.loc[posts['full_id'].isin(bully_posts['full_id']), 'class'] = 3
# else:
#     posts['class'] = 3
#     posts = posts[posts['full_id'].isin(bully_posts['full_id'])]
#
# posts['text'] = posts['text'].apply(extract_message)
# posts['text'] = posts['text'].apply(", ".join)
# posts = posts[['class', 'text']]
#
# posts.to_csv(os.path.join("data", "31.csv"))


# Labels each class multiple times
#### jigsaw_toxic
# source_path = os.path.join("data", "jigsaw-toxic_uo.csv")
# classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
# class_ids = {c: SPEECH_CLASSES.index("unclassified offensive") if c not in SPEECH_CLASSES else SPEECH_CLASSES.index(c)
#              for c in classes}
# column_names = ['text']
# column_names.extend(classes)
# toxic_posts: pd.DataFrame = pd.read_csv(source_path, header=0, usecols=[1, 2, 3, 4, 5, 6, 7],
#                                         names=column_names, dtype=str)
# print(toxic_posts)
# single_class_dataframes = list()
# for c in classes:
#     posts = toxic_posts[['text', c]]
#     posts = posts[posts[c] != '0']
#     posts['class'] = class_ids[c]
#     posts = posts[['class', 'text']]
#     single_class_dataframes.append(posts)
# total = pd.concat(single_class_dataframes, ignore_index=True)
# print(total)
# total.to_csv(os.path.join("data", "jigsaw-toxic.csv"))

