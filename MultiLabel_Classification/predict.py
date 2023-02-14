import torch
import utils
import numpy as np
from attention.model_add_cnn import StructuredSelfAttention
import data_got
from tqdm import tqdm
import argparse
import warnings

warnings.filterwarnings("ignore")


def predict(model_path, data_loader, gpu=True):
    """
    Predict function
    :param model: trained model
    :param path: model parameters file path
    :param data_loader: data used to predict
    :param gpu: True if use GPU, else False
    :return: a numpy array
    """

    config = utils.read_config("MultiLabel_Classification/config.yml")
    label_embed = torch.from_numpy(np.load("Word2Vec/label_embeddings.npy")).float()
    embed = torch.from_numpy(np.load("Word2Vec/embeddings.npy")).float()

    model = StructuredSelfAttention(
        batch_size=config.batch_size,
        lstm_hid_dim=config['lstm_hidden_dimension'],
        d_a=config["d_a"],
        n_classes=config.n_classes,
        label_embed=label_embed,
        embeddings=embed,
        embedding_size=config.embed_size,
        kernel_size=config.kernel_size,
        kernel_num=config.kernel_num
    )

    if gpu:
        model.cuda()

    model.load(model_path)
    model.eval()

    for idx, data in enumerate(tqdm(data_loader)):
        x = data[0]
        if gpu:
            x = x.cuda()
        with torch.no_grad():
            y_pred = model(x)
        y_pred = y_pred.cpu().numpy()
        label_mat = prob2label(y_pred, threshold=0.5)
        if idx == 0:
            rtn = label_mat
        else:
            rtn = np.concatenate([rtn, label_mat])

    return rtn


def prob2label(y_pred, threshold=0.5):
    """
    Convert predicted probability to label
    """
    score_mat = (y_pred >= threshold)*1
    return score_mat


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", help="path of the model file", default=None)
    args = parser.parse_args()

    train_loader, dev_loader, test_loader, \
        label_embed, embed, word_to_id = data_got.load_data(batch_size=128, drop_last_batch=False, shuffle=False)

    y_train_pred = predict(args.model_path, train_loader, gpu=True)
    y_dev_pred = predict(args.model_path, dev_loader, gpu=True)
    y_test_pred = predict(args.model_path, test_loader, gpu=True)
    np.save("MultiLabel_Classification/data/y_train_pred", y_train_pred)
    np.save("MultiLabel_Classification/data/y_dev_pred", y_dev_pred)
    np.save("MultiLabel_Classification/data/y_test_pred", y_test_pred)
