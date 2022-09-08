import numpy as np  # linear algebra
import torch
from collections import Counter
import matplotlib.pyplot as plt


def cuda_or_not():
    is_cuda = torch.cuda.is_available()

    # If we have a GPU available, we'll set our device to GPU. We'll use this device variable later in our code.
    if is_cuda:
        device = torch.device("cuda")
        print("GPU is available")
    else:
        device = torch.device("cpu")
        print("GPU not available, CPU used")

    return device


def preprocess(df, sample_frac=1):
    _df = df[['user_id', 'path']].groupby('user_id')['path'].apply(list)
    dd_df = df.drop_duplicates('user_id')
    dd_df.drop('path', axis=1, inplace=True)
    preprocessed_df = dd_df.merge(_df, on='user_id')
    preprocessed_df['path'] = preprocessed_df['path'].apply(lambda x: x[:500])
    preprocessed_df = preprocessed_df.sample(frac=sample_frac)

    return preprocessed_df


def tockenize(x_train, y_train, x_val, y_val):
    page_list = []

    for path in x_train:
        for page in path:
            page_list.append(page)

    corpus = Counter(page_list)
    # sorting on the basis of most common words
    corpus_ = sorted(corpus, key=corpus.get, reverse=True)  # [:1000]
    # creating a dict
    onehot_dict = {p: i+1 for i, p in enumerate(corpus_)}

    # tockenize
    final_list_train, final_list_test = [], []
    for path in x_train:
        final_list_train.append([onehot_dict[page] for page in path
                                 if page in onehot_dict.keys()])
    for path in x_val:
        final_list_test.append([onehot_dict[page] for page in path
                                if page in onehot_dict.keys()])

    encoded_train = [1 if label == 'm' else 0 for label in y_train]
    encoded_test = [1 if label == 'm' else 0 for label in y_val]

    return np.array(final_list_train), np.array(encoded_train), np.array(final_list_test), np.array(encoded_test), onehot_dict


def padding(paths, seq_len):
    features = np.zeros((len(paths), seq_len), dtype=int)
    for ii, review in enumerate(paths):
        if len(review) != 0:
            features[ii, -len(review):] = np.array(review)[:seq_len]

    return features

# function to predict accuracy


def acc(pred, label):
    pred = torch.round(pred.squeeze())

    return torch.sum(pred == label.squeeze()).item()


def visualize(epoch_tr_acc, epoch_vl_acc, epoch_tr_loss, epoch_vl_loss):
    fig = plt.figure(figsize=(20, 6))
    plt.subplot(1, 2, 1)
    plt.plot(epoch_tr_acc, label='Train Acc')
    plt.plot(epoch_vl_acc, label='Validation Acc')
    plt.title("Accuracy")
    plt.legend()
    plt.grid()

    plt.subplot(1, 2, 2)
    plt.plot(epoch_tr_loss, label='Train loss')
    plt.plot(epoch_vl_loss, label='Validation loss')
    plt.title("Loss")
    plt.legend()
    plt.grid()

    plt.show()
