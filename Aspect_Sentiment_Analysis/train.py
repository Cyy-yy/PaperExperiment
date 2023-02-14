import torch
from transformers import AutoTokenizer
from transformers.optimization import AdamW
from transformers import get_linear_schedule_with_warmup
from model import MainModule
# from model_wo_weighted_embedding import MainModule
# from model_wo_attention import MainModule
from sklearn.metrics import classification_report
import numpy as np
from tqdm import tqdm
from utils import save_state_dict
import argparse
import json
import os

parser = argparse.ArgumentParser()
parser.add_argument("--model_path", help="File path to save model", default=None)
parser.add_argument("--result_path", help="File path to save result", default=None)
args = parser.parse_args()


def loss_func(y_pred, target):
    """
    Loss function for multi-output classification
    :param y_pred, target: size=(batch_size, 18, 4)
    """
    target = target.float()
    num_samples = target.size(0)
    num_dimensions = target.size(1)
    log_pred = torch.log(y_pred)
    loss = -(target * log_pred).sum() / (num_dimensions * num_samples)
    return loss


def metrics(y_pred, y_true):
    """
    Calculate precision, recall and f1-score
    :param y_pred: size=(batch_size, 18, 4)
    :param y_true: size=(batch_size, 18, 4), one-hot labels
    """
    batch_size = y_pred.size(0)
    y_pred, y_true = y_pred.cpu().detach().numpy(), y_true.cpu().numpy()
    pred_label = y_pred.argmax(axis=-1)
    true_label = np.argwhere(y_true == 1)[:, 2].reshape([batch_size, 18])
    precision_lst, recall_lst, f1_lst = [], [], []
    for i in range(18):
        rtn = classification_report(
            true_label[:, i], pred_label[:, i], digits=4, output_dict=True
        )
        macro_avg = rtn["macro avg"]
        precision_lst.append(macro_avg["precision"])
        recall_lst.append(macro_avg["recall"])
        f1_lst.append(macro_avg["f1-score"])
    precision = np.mean(precision_lst)
    recall = np.mean(recall_lst)
    f1_score = np.mean(f1_lst)
    return precision, recall, f1_score


def train(train_loader, dev_loader, embeddings,
          epochs, lr, warmup_rate, pool_prop):
    """
    Main training function
    """
    tokenizer = AutoTokenizer.from_pretrained("nghuyong/ernie-1.0",
                                              model_max_length=512)

    model = MainModule(embeddings, pool_prop)
    model.cuda()

    num_training_steps = epochs * len(train_loader)
    num_warmup_steps = int(warmup_rate * num_training_steps)
    optimizer = AdamW(model.parameters(), lr=lr)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )
    # scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(
    #     optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps,
    #     num_cycles=epochs//2
    # )

    start_epoch = 0

    """Continue training"""
    if os.path.exists(args.model_path):
        print()
        print("Loading Checkpoints ... ...")
        checkpoint = torch.load(args.model_path)
        start_epoch = checkpoint["start_epoch"]
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["opt"])
        scheduler.load_state_dict(checkpoint["lr_scheduler"])
        print("Checkpoints Loaded !")
        print()

    res_path = args.result_path
    if os.path.exists(res_path):
        with open(res_path, "r") as fp:
            res = json.load(fp)
    else:
        res = dict()

    for epoch in range(start_epoch, epochs):

        res[f"epoch_{epoch + 1}"] = dict()

        """Traning stage"""

        model.train()
        print()
        print(
            f"Running EPOCH {epoch + 1}    lr = {optimizer.param_groups[0]['lr']:.8f}")
        print()
        train_loss = []
        precision = []
        recall = []
        f1_score = []

        for batch_idx, batch_data in enumerate(tqdm(train_loader)):
            optimizer.zero_grad()
            word_index, dim_index, dim_attention_mask, sentiment_label = \
                batch_data["word_index"].cuda(), \
                batch_data["dim_index"].cuda(), \
                batch_data["dim_attention_mask"].cuda(), \
                batch_data["sentiment_label"].cuda()
            review = batch_data["review"]
            token_ids = tokenizer(review, padding="max_length", truncation=True)
            token_ids = {
                key: torch.LongTensor(value).cuda() for (key, value) in
                token_ids.items()
            }
            y_pred = model(
                **token_ids, word_index=word_index, dim_index=dim_index,
                dim_attention_mask=dim_attention_mask
            )
            loss = loss_func(y_pred, sentiment_label)
            loss.backward()
            optimizer.step()
            scheduler.step()
            train_loss.append(loss.item())
            p, r, f1 = metrics(y_pred, sentiment_label)
            precision.append(p)
            recall.append(r)
            f1_score.append(f1)

        epoch_loss = np.mean(train_loss)
        epoch_p = np.mean(precision)
        epoch_r = np.mean(recall)
        epoch_f1 = np.mean(f1_score)

        res[f"epoch_{epoch + 1}"]["train_loss"] = epoch_loss
        res[f"epoch_{epoch + 1}"]["train_p"] = epoch_p
        res[f"epoch_{epoch + 1}"]["train_r"] = epoch_r
        res[f"epoch_{epoch + 1}"]["train_f1"] = epoch_f1

        print(f"EPOCH {epoch + 1} Train End : avg_loss = {epoch_loss:.4f}")
        print(
            f"precision: {epoch_p:.4f}, recall: {epoch_r:.4f}, f1-score:{epoch_f1:.4f}")
        print()

        """Evaluation Stage"""
        model.eval()
        val_loss = []
        val_precision = []
        val_recall = []
        val_f1 = []

        for batch_idx, batch_data in enumerate(tqdm(dev_loader)):
            word_index, dim_index, dim_attention_mask, sentiment_label = \
                batch_data["word_index"].cuda(), \
                batch_data["dim_index"].cuda(), \
                batch_data["dim_attention_mask"].cuda(), \
                batch_data["sentiment_label"].cuda()
            review = batch_data["review"]
            token_ids = tokenizer(review, padding="max_length", truncation=True)
            token_ids = {
                key: torch.LongTensor(value).cuda() for (key, value) in
                token_ids.items()
            }
            with torch.no_grad():
                y_pred = model(
                    **token_ids, word_index=word_index, dim_index=dim_index,
                    dim_attention_mask=dim_attention_mask
                )
                loss = loss_func(y_pred, sentiment_label)
                p, r, f1 = metrics(y_pred, sentiment_label)
            val_loss.append(loss.item())
            val_precision.append(p)
            val_recall.append(r)
            val_f1.append(f1)

        epoch_val_loss = np.mean(val_loss)
        epoch_val_p = np.mean(val_precision)
        epoch_val_r = np.mean(val_recall)
        epoch_val_f1 = np.mean(val_f1)

        res[f"epoch_{epoch + 1}"]["val_loss"] = epoch_val_loss
        res[f"epoch_{epoch + 1}"]["val_p"] = epoch_val_p
        res[f"epoch_{epoch + 1}"]["val_r"] = epoch_val_r
        res[f"epoch_{epoch + 1}"]["val_f1"] = epoch_val_f1

        print(
            f"EPOCH {epoch + 1} Evaluation End : avg_loss = {epoch_val_loss:.4f}")
        print(
            f"precision: {epoch_val_p:.4f}, recall: {epoch_val_r:.4f}, f1-score:{epoch_val_f1:.4f}")

        if (args.model_path != "") and ((epoch + 1) % 1 == 0):
            print()
            save_state_dict(model, optimizer, scheduler, epoch, args.model_path)

        print()
        print("=" * 60)

        with open(res_path, "w", encoding="utf-8") as fp:
            json.dump(res, fp, ensure_ascii=False, indent=4)
