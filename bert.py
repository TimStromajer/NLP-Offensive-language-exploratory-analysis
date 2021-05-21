import torch
from transformers import BertTokenizer, BertModel
from scipy.spatial.distance import cosine
import pickle
import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
import numpy as np

from bert_vectors import mean, sent, mean2, mean3
from speech_classes import SPEECH_CLASSES
from dense_plotting import format_label


def bert_to_vec(text, model, tokenizer, class_word=None):
    if class_word is not None:
        # text = "Here is the sentence I want embeddings for."
        marked_text = "[CLS] " + text + " [SEP]"

        # Tokenize our sentence with the BERT tokenizer.
        tokenized_text = tokenizer.tokenize(marked_text)

        # Tokenize class sentence with the BERT tokenizer.
        tokenized_class = tokenizer.tokenize(class_word)

    # # Map the token strings to their vocabulary indeces.
    # indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    #
    # # Mark each of the 22 tokens as belonging to sentence "1".
    # segments_ids = [1] * len(tokenized_text)
    #
    # # Convert inputs to PyTorch tensors
    # tokens_tensor = torch.tensor([indexed_tokens])
    # segments_tensors = torch.tensor([segments_ids])

    # Encode the sentence
    encoded = tokenizer.encode_plus(
        text=text,  # the sentence to be encoded
        add_special_tokens=True,  # Add [CLS] and [SEP]
        #max_length=64,  # maximum length of a sentence
        padding=True,  # Add [PAD]s
        return_attention_mask=True,  # Generate the attention mask
        return_tensors='pt',  # ask the function to return PyTorch tensors
    )

    # Get the input IDs and attention mask in tensor format
    tokens_tensor = encoded['input_ids']
    segments_tensors = encoded['attention_mask']

    # Run the text through BERT, and collect all of the hidden states produced
    # from all 12 layers.
    with torch.no_grad():
        outputs = model(tokens_tensor, segments_tensors)

        # Evaluating the model will return a different number of objects based on
        # how it's  configured in the `from_pretrained` call earlier. In this case,
        # becase we set `output_hidden_states = True`, the third item will be the
        # hidden states from all layers. See the documentation for more details:
        # https://huggingface.co/transformers/model_doc/bert.html#bertmodel
        hidden_states = outputs[2]

    # print("Number of layers:", len(hidden_states), "  (initial embeddings + 12 BERT layers)")
    # layer_i = 0
    #
    # print("Number of batches:", len(hidden_states[layer_i]))
    # batch_i = 0
    #
    # print("Number of tokens:", len(hidden_states[layer_i][batch_i]))
    # token_i = 0
    #
    # print("Number of hidden units:", len(hidden_states[layer_i][batch_i][token_i]))

    if class_word is not None:
        tokens = len(hidden_states[0][0])
        start = tokens - 2 - len(tokenized_class)
        stop = -2

    # Concatenate the tensors for all layers. We use `stack` here to
    # create a new dimension in the tensor.
    token_embeddings = torch.stack(hidden_states, dim=0)

    # Remove dimension 1, the "batches".
    token_embeddings = torch.squeeze(token_embeddings, dim=1)

    # Swap dimensions 0 and 1.
    token_embeddings = token_embeddings.permute(1, 0, 2)

    ### concatinating layers
    # token_vecs_cat = []
    #
    # # For each token in the sentence...
    # for token in token_embeddings:
    #     # `token` is a [12 x 768] tensor
    #
    #     # Concatenate the vectors (that is, append them together) from the last
    #     # four layers.
    #     # Each layer vector is 768 values, so `cat_vec` is length 3,072.
    #     cat_vec = torch.cat((token[-1], token[-2], token[-3], token[-4]), dim=0)
    #
    #     # Use `cat_vec` to represent `token`.
    #     token_vecs_cat.append(cat_vec)
    # return torch.mean(torch.stack(token_vecs_cat)) <- wrong

    ### summing layers
    # token_vecs_sum = []
    #
    # # `token_embeddings` is a [22 x 12 x 768] tensor.
    #
    # # For each token in the sentence...
    # for token in token_embeddings:
    #     # `token` is a [12 x 768] tensor
    #
    #     # Sum the vectors from the last four layers.
    #     sum_vec = torch.sum(token[-4:], dim=0)
    #
    #     # Use `sum_vec` to represent `token`.
    #     token_vecs_sum.append(sum_vec)
    #
    # return torch.mean(torch.stack(token_vecs_sum)) <- wrong

    # `token_vecs` is a tensor with shape [22 x 768]
    token_vecs = hidden_states[-2][0]

    if class_word is None:
        # Calculate the average of all token vectors.
        sentence_embedding = torch.mean(token_vecs, dim=0)
    else:
        # Calculate the average of all class token vectors.
        sentence_embedding = torch.mean(token_vecs[start:stop], dim=0)

    return sentence_embedding


def load_model_and_tokenizer():
    # Load pre-trained model tokenizer (vocabulary)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # Load pre-trained model (weights)
    model = BertModel.from_pretrained('bert-base-uncased',
                                      output_hidden_states=True,  # Whether the model returns all hidden-states.
                                      )

    # Put the model in "evaluation" mode, meaning feed-forward operation.
    model.eval()
    return model, tokenizer


def get_texts(tables, chosen_class, max_len=100000, added_sent=False):
    texts = []
    for t in tables:
        table_data = pd.read_csv("data/" + t)
        for idx, row in enumerate(table_data.iterrows()):
            id = row[1]["class"]
            text = row[1]["text"]
            if id == chosen_class:
                if added_sent:
                    if len(text) > 480: #max is 512
                        texts.append((text[:480] + str(". This is " + SPEECH_CLASSES[chosen_class] + ".")))
                    else:
                        if text[-1] != ".":
                            texts.append((text + str(". This is " + SPEECH_CLASSES[chosen_class] + ".")))
                        else:
                            texts.append((text + str(" This is " + SPEECH_CLASSES[chosen_class] + ".")))
                else:
                    if len(text) > 512:
                        texts.append(text[:512])
                    else:
                        texts.append(text)
            if len(texts) >= max_len:
                break
    print("number of texts:", len(texts))
    return texts


def get_mean_and_sentence(model, tokenizer, texts):
    embeddings = []
    for t in texts:
        embd = bert_to_vec(t, model, tokenizer)
        embeddings.append(embd)

    embd_mean = torch.mean(torch.stack(embeddings), dim=0)

    cos_from_mean = []
    for i, e in enumerate(embeddings):
        cos = cosine_dist(e, embd_mean)
        cos_from_mean.append(cos)
    max_cos = cos_from_mean.index(max(cos_from_mean))

    return embd_mean, texts[max_cos]


def get_class_mean_and_sentence(model, tokenizer, texts, chosen_class, only_class_word=False):
    embeddings = []
    for t in texts:
        if only_class_word:
            embd = bert_to_vec(t, model, tokenizer, SPEECH_CLASSES[chosen_class])
        else:
            embd = bert_to_vec(t, model, tokenizer)
        embeddings.append(embd)

    embd_mean = torch.mean(torch.stack(embeddings), dim=0)

    cos_from_mean = []
    for i, e in enumerate(embeddings):
        cos = cosine_dist(e, embd_mean)
        cos_from_mean.append(cos)
    max_cos = cos_from_mean.index(max(cos_from_mean))

    return embd_mean, texts[max_cos]


def cosine_dist(vec1, vec2):
    return 1 - cosine(vec1, vec2)


def visualize_dendrogram(mean_):
    dists = []
    labels = []
    for i, m1 in enumerate(mean_):
        for j, m2 in enumerate(mean_):
            if j > i:
                dist = cosine(m1, m2)
                if dist > 0:
                    dists.append(dist)
        if len(m1) > 1:
            labels.append(SPEECH_CLASSES[i])

    Z = linkage(dists, 'average')
    fig = plt.figure(figsize=(15, 8))
    labels_formatted = [format_label(label) for label in labels]
    dn = dendrogram(Z, labels=labels_formatted, leaf_rotation=25)
    plt.show()


def calculate_bert_vectors(tables, chosen_class, added_sent=False, only_class_word=False):
    model, tokenizer = load_model_and_tokenizer()
    texts = get_texts(tables, chosen_class, added_sent=added_sent)
    class_mean, sentence = get_class_mean_and_sentence(model, tokenizer, texts, chosen_class, only_class_word=only_class_word)
    print(sentence)
    print(class_mean)
    return class_mean

if __name__ == '__main__':

    # mean and sentences saved in bert_vectors.py
    # tables = ['9.csv', '21.csv', '25.csv', '26.csv', '31.csv', '32.csv', 'jigsaw-toxic.csv']
    # choosen_class = [1, 2, 3, 5, 6, 7, 10, 11, 13, 14, 16, 17, 18, 19, 20, 22, 23, 24]
    # calc_mean = [[0] for _ in range(25)]
    # for c in choosen_class:
    #     calc_mean = calculate_bert_vectors(tables, c, added_sent=True, only_class_word=True)
    #     calc_mean[c] = calc_mean

    visualize_dendrogram(mean)
    visualize_dendrogram(mean3)




