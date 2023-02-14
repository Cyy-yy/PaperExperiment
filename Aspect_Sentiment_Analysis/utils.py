import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import yaml


class MyDataset(Dataset):

    def __init__(self,
                 review_path,
                 dim_path,
                 word_index_path,
                 dim_attention_mask_path,
                 sentiment_label_path):
        super(MyDataset, self).__init__()
        self.review = pd.read_csv(review_path)["review_tokenized"]
        self.review = self.review.apply(lambda x: "".join(eval(x))).tolist()
        self.dim = np.load(dim_path)
        self.word_index = np.load(word_index_path)
        self.dim_attention_mask = np.load(dim_attention_mask_path)
        self.sentiment_label = np.load(sentiment_label_path)

    def __len__(self):
        return len(self.review)

    def __getitem__(self, idx):
        review = self.review[idx]
        dim = self.dim[idx]
        word_index = self.word_index[idx]
        dim_attention_mask = self.dim_attention_mask[idx]
        sentiment_label = self.sentiment_label[idx]
        return {
            "review": review,
            "dim_index": torch.LongTensor(dim),
            "word_index": torch.LongTensor(word_index),
            "dim_attention_mask": torch.LongTensor(dim_attention_mask),
            "sentiment_label": torch.LongTensor(sentiment_label)
        }


def get_dataloader(dataset, batch_size=8, drop_last=True, shuffle=True):
    data_loader = DataLoader(dataset, batch_size, shuffle=shuffle, drop_last=drop_last)
    return data_loader


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def get_config(path):
    with open(path, "r") as fp:
        config = yaml.safe_load(fp)
    return AttrDict(config)


def save_state_dict(model, opt, lr_scheduler, epoch, path):
    print("Saving Files ... ...")
    state_dict = {
        "start_epoch": epoch+1,
        "model": model.state_dict(),
        "opt": opt.state_dict(),
        "lr_scheduler": lr_scheduler.state_dict()
    }
    try:
        torch.save(state_dict, path)
        print("Files Saved !")
    except Exception as e:
        print("Wrong Saving ! ! !")
        print(e)
