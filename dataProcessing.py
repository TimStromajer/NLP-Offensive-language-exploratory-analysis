import pandas as pd
import sqlite3
from sqlite3 import OperationalError
from sqlalchemy import create_engine
import json

##### 9
# tab = pd.read_csv("data/9_uo.csv")
# tab2 = tab.replace({'class': {0: 13, 2: 0}})    # 1 -> 1
# tab2 = tab2[["class", "tweet"]]
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

