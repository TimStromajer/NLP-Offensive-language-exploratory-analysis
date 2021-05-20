# NLP-Offensive-language-exploratory-analysis

## Reproduction of the results
In order the obtain the same results as we have in our explanatory analysis, you must prepare the same data.
### Data
Data was obtained from numerous data sources and then processed into the same shape. Some of the Already processed data  is available [here](https://github.com/TimStromajer/NLP-Offensive-language-exploratory-analysis/tree/main/data) (only those that are publicly available). The script, that process all the data, is available [here](https://github.com/TimStromajer/NLP-Offensive-language-exploratory-analysis/blob/main/text_processing.py). Here are listed all used data sources and how we processed them.

 - [Automated Hate Speech Detection and the Problem of Offensive Language](https://github.com/t-davidson/hate-speech-and-offensive-language)
 From the file *labeled_data.csv* we extracted tweet and class and created a new table.
 Included classes are: offensive and hateful.
	> Already processed [9.csv](https://github.com/TimStromajer/NLP-Offensive-language-exploratory-analysis/blob/main/data/9.csv)

 - [Exploring Hate Speech Detection in Multimodal Publications](https://gombru.github.io/2019/10/09/MMHS/)
 From the file *MMHS150K_GT.json* we extracted labels and tweet texts. If at least two out of three labels were the same, we kept the tweet with corresponding class, if not we skipped it.
Included classes are: racist, homophobic, sexist and religion.
	> Already processed [21.csv](https://github.com/TimStromajer/NLP-Offensive-language-exploratory-analysis/blob/main/data/21.csv)
	
- [HASOC](https://hasocfire.github.io/hasoc/2019/dataset.html)
From the file *english_dataset.tsv* we extracted text and task_2 (class) and created a new table.
Included classes are: offensive, profane and hateful.
	> Already processed [25.csv](https://github.com/TimStromajer/NLP-Offensive-language-exploratory-analysis/blob/main/data/25.csv)
	
- [Large Scale Crowdsourcing and Characterization of Twitter Abusive Behavior](https://github.com/ENCASEH2020/hatespeech-twitter)
From the file *hatespeech_labels.csv* we extracted tweet ids with abusive label and then obtained all tweets with Tweeter API.
Included classes: abusive.
	> Not publicly available. To get the data, please contact us.
	
### Results
#### Bert
You can obtain BERT results by running script [*bert.py*](https://github.com/TimStromajer/NLP-Offensive-language-exploratory-analysis/blob/main/bert.py) (function called *visualize_dendrogram*). Due to long calculations, BERT vectors are already calculated and stored in [bert_vectors.py](https://github.com/TimStromajer/NLP-Offensive-language-exploratory-analysis/blob/main/bert_vectors.py). But if you would like to calculate it yourself, you can call the function *calculate_bert_vectors*. This will again calculate BERT vectors, which can be then used in function *visualize_dendrogram*. 