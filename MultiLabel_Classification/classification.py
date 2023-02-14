from attention.model import StructuredSelfAttention
# from attention.model_add_cnn import StructuredSelfAttention
from attention.train import train
import torch
from torch.optim.lr_scheduler import StepLR
import utils
import data_got
from tqdm import tqdm
import numpy as np
import json

# load configuration file in `dict` type
config = utils.read_config("MultiLabel_Classification/config.yml")
label_num = config.n_classes

if config.GPU:
    torch.cuda.set_device(0)

print('loading data...')

train_loader, dev_loader, test_loader, \
label_embed, embed, word_to_id = data_got.load_data(
    batch_size=config.batch_size)  # laod dataset in PyTorch DataLoader

# load pretrained embeddings
label_embed = torch.from_numpy(label_embed).float()  # [L*768]
embed = torch.from_numpy(embed).float()

print("load done")


def multilabel_classification(attention_model, train_loader, dev_loader,
                              test_loader, epochs, lr, gpu=True, test=False):
    """
    Main training function
    """
    loss = torch.nn.BCELoss()  # Binary Cross Entropy Loss Function
    opt = torch.optim.Adam(attention_model.parameters(), lr=lr,
                           betas=(0.9, 0.99))  # use Adam optimizer
    scheduler = StepLR(opt, step_size=config.decay_per_step,
                       gamma=config.decay_rate)
    train(
        attention_model, train_loader, dev_loader, loss, opt, scheduler,
        label_num, epochs, gpu, test=test
    )  # return nothing, just output some training and evaluation info


# LSAN
attention_model = StructuredSelfAttention(
    batch_size=config.batch_size,
    lstm_hid_dim=config['lstm_hidden_dimension'],
    d_a=config["d_a"],
    n_classes=label_num,
    label_embed=label_embed,
    embeddings=embed
)


# CNN-LSAN
# attention_model = StructuredSelfAttention(
#     batch_size=config.batch_size,
#     lstm_hid_dim=config['lstm_hidden_dimension'],
#     d_a=config["d_a"],
#     n_classes=label_num,
#     label_embed=label_embed,
#     embeddings=embed,
#     embedding_size=config.embed_size,
#     kernel_size=config.kernel_size,
#     kernel_num=config.kernel_num
# )

if config.use_cuda:
    attention_model.cuda()

multilabel_classification(
    attention_model, train_loader, dev_loader, test_loader,
    epochs=config["epochs"], lr=config["learning_rate"],
    test=True
)
