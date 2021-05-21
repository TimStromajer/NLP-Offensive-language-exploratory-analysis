# NLP-Offensive-language-exploratory-analysis
Public [repository](https://github.com/TimStromajer/NLP-Offensive-language-exploratory-analysis-project) with the private data removed:
## Installing dependencies
Run the command ```pip install -r requirements.txt``` to install the dependencies.

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

- [A Qality Type-aware Annotated Corpus and Lexicon for Harrasment Research](https://github.com/Mrezvan94/Harassment-Corpus)
The individual files were labeled based on the file-name and merged. Only positive instances were kept, because each file contains a different corpus of tweets and only labels for a specific class. If the tweet belonged to a different offensive class, it was labeled as inoffensive.
Included classes are: racist, sexist, appearence-related, intelligance and political harassment.
	> Not publicly available.
	
- [Detecting cyberbullying in online communities (League of Legends)](http://ub-web.de/research/)
The sql dump was converted to csv using [mysqldump_to_csv.py](https://github.com/jamesmishra/mysqldump-to-csv).
	> Already processed [25.csv](https://github.com/TimStromajer/NLP-Offensive-language-exploratory-analysis/blob/main/data/31.csv)

- [Toxic Comment Classification Challenge](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/data)
The train.csv was used as is. test.csv and test_labels.csv was manually merged to resemble the same format. The files should be named *jigsaw-toxic_test_uo.csv* and *jigsaw-toxic_train_uo.csv* and placed in the *data/* folder. The data processing script combines them and converts to the expected format. (Uncomment the appropriate part)
	> Available with a Kaggle account at the provided link
	
### Results
#### Vocabulary analysis with non-contextual denseembedding
The results are obtained by running the script [*w2v_term_analysis.py*](https://github.com/TimStromajer/NLP-Offensive-language-exploratory-analysis/blob/main/w2v_term_analysis.py). The script will download all the required models on first runtime. All intermediate results and final visualizations are saved to the *w2v_term_analysis* directory. To re-run all the calculations, remove all contents from the folder.
#### Representative documents for each class
The results are obtained by running the script [*w2v_document_embeddings.py*](https://github.com/TimStromajer/NLP-Offensive-language-exploratory-analysis/blob/main/w2v_document_embeddings.py). The script will download all the required models on first runtime. Intermediate data is saved to the *w2v_document_embeddings.py-intermediate_data/* directory. The final results are printed to console.
#### Bert
You can obtain BERT results by running script [*bert.py*](https://github.com/TimStromajer/NLP-Offensive-language-exploratory-analysis/blob/main/bert.py) (function called *visualize_dendrogram*). Due to long calculations, BERT vectors are already calculated and stored in [bert_vectors.py](https://github.com/TimStromajer/NLP-Offensive-language-exploratory-analysis/blob/main/bert_vectors.py). But if you would like to calculate it yourself, you can call the function *calculate_bert_vectors*. This will again calculate BERT vectors, which can be then used in function *visualize_dendrogram*. 
#### K-means
You can obtain the k-means results by running the script [*kmeans.py*](https://github.com/TimStromajer/NLP-Offensive-language-exploratory-analysis/blob/main/kmeans.py). Upon first run, all models and documents are saved to improve performance of subsequent runs. If you wish to rerun it with different parameters, delete the folders ```data_pickles``` and ```model_pickles```. The graph is saved in the file ```kmeans.png```.

### Offensive language schema
![Offensive language schema](schema/Schema.png)
