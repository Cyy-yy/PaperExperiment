import numpy as np
import pandas as pd
import torch.utils.data as data_utils
import torch


def load_data(batch_size=64, drop_last_batch=True, shuffle=True):
    """
    :return embed.idx_to_vec.asnumpy(): size=(vocab_size, embedding_size)
    :return embed.token_to_idx: a {word: idx} dictionary
    """
    X_tst = np.load("MultiLabel_Classification/data/X_test.npy")
    X_trn = np.load("MultiLabel_Classification/data/X_train.npy")
    X_dev = np.load("MultiLabel_Classification/data/X_dev.npy")
    Y_trn = np.load("MultiLabel_Classification/data/y_train.npy")
    Y_tst = np.load("MultiLabel_Classification/data/y_test.npy")
    Y_dev = np.load("MultiLabel_Classification/data/y_dev.npy")
    label_embeddings = np.load("Word2Vec/label_embeddings.npy")
    word_embeddings = np.load("Word2Vec/embeddings.npy")
    word_index = dict()
    with open("Word2Vec/vocab.txt", "r") as fp:
        vocab = fp.read().split("\n")
        for idx, w in enumerate(vocab):
            word_index[w] = idx
    train_data = data_utils.TensorDataset(torch.from_numpy(X_trn).type(torch.LongTensor),
                                          torch.from_numpy(Y_trn).type(torch.LongTensor))
    test_data = data_utils.TensorDataset(torch.from_numpy(X_tst).type(torch.LongTensor),
                                         torch.from_numpy(Y_tst).type(torch.LongTensor))
    dev_data = data_utils.TensorDataset(torch.from_numpy(X_dev).type(torch.LongTensor),
                                         torch.from_numpy(Y_dev).type(torch.LongTensor))
    train_loader = data_utils.DataLoader(train_data, batch_size, shuffle=shuffle, drop_last=drop_last_batch)
    dev_loader = data_utils.DataLoader(dev_data, batch_size, shuffle=False, drop_last=False)
    test_loader = data_utils.DataLoader(test_data, batch_size, shuffle=False, drop_last=False)
    # return train_loader, test_loader, label_embed, embed.idx_to_vec.asnumpy(), X_tst, embed.token_to_idx, Y_tst, Y_trn
    return train_loader, dev_loader, test_loader, label_embeddings, word_embeddings, word_index

