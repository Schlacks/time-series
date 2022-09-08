import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
import pickle
import configparser

from utils import cuda_or_not, preprocess, tockenize, padding, acc, visualize
from models import CatSeqRNN

if __name__ == "__main__":

    config = configparser.ConfigParser()
    config.read('../config.ini')
    data_path = config['LSTM_SHALLOW']['data_path']
    sample_frac = float(config['LSTM_SHALLOW']['sample_frac'])
    batch_size = int(config['LSTM_SHALLOW']['batch_size'])
    no_layers = int(config['LSTM_SHALLOW']['no_layers'])
    embedding_dim = int(config['LSTM_SHALLOW']['embedding_dim'])
    hidden_dim = int(config['LSTM_SHALLOW']['hidden_dim'])
    lr = float(config['LSTM_SHALLOW']['lr'])
    epochs = int(config['LSTM_SHALLOW']['epochs'])

    device = cuda_or_not()

    df = pd.read_csv(data_path)
    df = preprocess(df, sample_frac=sample_frac)

    X, y = df['path'].values, df['gender'].values
    x_train, x_test, y_train, y_test = train_test_split(X, y, stratify=y)
    print(f'shape of train data is {x_train.shape}')
    print(f'shape of test data is {x_test.shape}')

    x_train, y_train, x_test, y_test, page_vol = tockenize(
        x_train, y_train, x_test, y_test)

    # save tokenization dictionary for reference during future model application
    with open('page_token_dict.pkl', 'wb') as f:
        pickle.dump(page_vol, f)

    x_train_pad = padding(x_train, 500)
    x_test_pad = padding(x_test, 500)

    # create Tensor datasets
    train_data = TensorDataset(torch.from_numpy(
        x_train_pad), torch.from_numpy(y_train))
    valid_data = TensorDataset(torch.from_numpy(
        x_test_pad), torch.from_numpy(y_test))

    # make sure to SHUFFLE your data
    train_loader = DataLoader(train_data, shuffle=True,
                              batch_size=batch_size, drop_last=True)
    valid_loader = DataLoader(valid_data, shuffle=True,
                              batch_size=batch_size, drop_last=True)

    # obtain one batch of training data
    dataiter = iter(train_loader)
    sample_x, sample_y = dataiter.next()

    print('Sample input size: ', sample_x.size())  # batch_size, seq_length
    print('Sample input: \n', sample_x)
    print('Sample input: \n', sample_y)

    vocab_size = len(page_vol) + 1  # extra 1 for padding
    output_dim = 1
    clip = 5
    model = CatSeqRNN(no_layers, vocab_size, hidden_dim,
                      embedding_dim, output_dim, drop_prob=0.5)

    # moving to gpu
    model.to(device)

    # loss and optimization functions

    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    valid_loss_min = np.Inf
    # train for some number of epochs
    epoch_tr_loss, epoch_vl_loss = [], []
    epoch_tr_acc, epoch_vl_acc = [], []

    for epoch in range(epochs):
        train_losses = []
        train_acc = 0.0
        model.train()
        # initialize hidden state
        h = model.init_hidden(batch_size, device)
        for inputs, labels in train_loader:

            inputs, labels = inputs.to(device), labels.to(device)
            # Creating new variables for the hidden state, otherwise
            # we'd backprop through the entire training history
            h = tuple([each.data for each in h])

            model.zero_grad()
            output, h = model(inputs, h)

            # calculate the loss and perform backprop
            loss = criterion(output.squeeze(), labels.float())
            loss.backward()
            train_losses.append(loss.item())
            # calculating accuracy
            accuracy = acc(output, labels)
            train_acc += accuracy
            # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
            nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()

        val_h = model.init_hidden(batch_size, device)
        val_losses = []
        val_acc = 0.0
        model.eval()
        for inputs, labels in valid_loader:
            val_h = tuple([each.data for each in val_h])

            inputs, labels = inputs.to(device), labels.to(device)

            output, val_h = model(inputs, val_h)
            val_loss = criterion(output.squeeze(), labels.float())

            val_losses.append(val_loss.item())

            accuracy = acc(output, labels)
            val_acc += accuracy

        epoch_train_loss = np.mean(train_losses)
        epoch_val_loss = np.mean(val_losses)
        epoch_train_acc = train_acc/len(train_loader.dataset)
        epoch_val_acc = val_acc/len(valid_loader.dataset)
        epoch_tr_loss.append(epoch_train_loss)
        epoch_vl_loss.append(epoch_val_loss)
        epoch_tr_acc.append(epoch_train_acc)
        epoch_vl_acc.append(epoch_val_acc)
        print(f'Epoch {epoch+1}')
        print(f'train_loss : {epoch_train_loss} val_loss : {epoch_val_loss}')
        print(
            f'train_accuracy : {epoch_train_acc*100} val_accuracy : {epoch_val_acc*100}')
        # if epoch_val_loss <= valid_loss_min:
        if True:
            #torch.save(model.state_dict(), 'state_dict.pt')
            torch.save(model, 'saved_model.pt')
            print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
                valid_loss_min, epoch_val_loss))
            valid_loss_min = epoch_val_loss
        print(25*'==')

visualize(epoch_tr_acc, epoch_vl_acc, epoch_tr_loss, epoch_vl_loss)
