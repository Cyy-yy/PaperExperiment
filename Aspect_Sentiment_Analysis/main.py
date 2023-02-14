from utils import get_dataloader, get_config
from utils import MyDataset
from train import train
import numpy as np


if __name__ == "__main__":

    config = get_config("config.yml")
    
    print("Loading Dataset ... ...")

    train_dataset = MyDataset(
        "../ASAP_clean/train.csv",
        "data/pred_dim_train.npy",
        "data/X_train_SA.npy",
        "data/dim_attention_mask_train.npy",
        "data/y_train.npy"
    )
    dev_dataset = MyDataset(
        "../ASAP_clean/dev.csv",
        "data/pred_dim_dev.npy",
        "data/X_dev_SA.npy",
        "data/dim_attention_mask_dev.npy",
        "data/y_dev.npy"
    )
    train_loader = get_dataloader(
        train_dataset, batch_size=config.batch_size, drop_last=True, shuffle=True
    )
    dev_loader = get_dataloader(
        dev_dataset, batch_size=config.batch_size, drop_last=False, shuffle=False
    )
    # embeddings = np.load("../Word2Vec/embeddings_trained.npy")
    embeddings = np.load("../Word2Vec/embeddings.npy")
    
    print("Dataset Loaded !")

    train(train_loader, dev_loader, embeddings, config.epochs, config.lr, config.warmup_rate, config.pool_prop)
