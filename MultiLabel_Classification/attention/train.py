import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from sklearn.metrics import f1_score, classification_report
import warnings
import argparse
import json

warnings.filterwarnings("ignore")


parser = argparse.ArgumentParser()
parser.add_argument("--save_path", help="File path to save model", default=None)
parser.add_argument("--predict_result_path", help="File path to save prediction result", default=None)
args = parser.parse_args()


def train(attention_model, train_loader, test_loader, criterion, opt, scheduler,
          num_class, epochs=5, gpu=True, test=False):
    """
    :criterion: loss function
    :opt: optimizer
    """
    if gpu:
        attention_model.cuda()

    for i in range(epochs):
        print()
        print("Running EPOCH", i+1, f"lr={opt.param_groups[0]['lr']:.5f}")
        print()
        """
        Training Stage
        """
        attention_model.train()
        train_loss = []  # store loss of each batch
        precision = []
        recall = []
        f1 = []
        for batch_idx, train_data in enumerate(tqdm(train_loader)):
            opt.zero_grad()
            x, y = train_data[0].cuda(), train_data[1].cuda()
            y_pred = attention_model(x)
            loss = criterion(y_pred, y.float())/train_loader.batch_size
            loss.backward()
            opt.step()
            scheduler.step()
            labels_cpu = y.data.cpu().float()
            pred_cpu = y_pred.data.cpu()
            macro_precision, macro_recall, macro_f1 = metrics(labels_cpu.numpy(), pred_cpu.numpy())
            precision.append(macro_precision)
            recall.append(macro_recall)
            f1.append(macro_f1)
            train_loss.append(float(loss))
        avg_loss = np.mean(train_loss)  # average loss of each epoch
        epoch_p = np.mean(precision)
        epoch_r = np.mean(recall)
        epoch_f = np.mean(f1)
        print("epoch %2d train end : avg_loss = %.4f" % (i+1, avg_loss))
        print("precision: %.4f, recall: %.4f, f1-score: %.4f" % (epoch_p, epoch_r, epoch_f))
        
        """
        Evalutation Stage
        """
        attention_model.eval()
        test_precision = []
        test_recall = []
        test_f1 = []
        test_loss = []
        print()
        for batch_idx, test in enumerate(tqdm(test_loader)):
            x, y = test[0].cuda(), test[1].cuda()
            with torch.no_grad():
                val_y = attention_model(x)
            loss = criterion(val_y, y.float()) / train_loader.batch_size
            labels_cpu = y.data.cpu().float()
            pred_cpu = val_y.data.cpu()
            macro_precision, macro_recall, macro_f1 = metrics(labels_cpu.numpy(), pred_cpu.numpy())
            test_precision.append(macro_precision)
            test_recall.append(macro_recall)
            test_f1.append(macro_f1)
            test_loss.append(float(loss))
        avg_test_loss = np.mean(test_loss)
        test_p = np.mean(test_precision)
        test_r = np.mean(test_recall)
        test_f = np.mean(test_f1)
        print("epoch %2d test end : avg_loss = %.4f" % (i+1, avg_test_loss))
        print("precision: %.4f, recall: %.4f, f1-score: %.4f" % (test_p, test_r, test_f))
        print()
        print("="*60)

    if test:
        test_stage(attention_model, test_loader, gpu, num_class)

    attention_model.save(args.save_path)


def test_stage(model, test_loader, gpu, num_class):
    model.eval()
    precision = []
    recall = []
    f1 = []
    label_metrics = {
        str(label): {"precision": [], "recall": [], "f1": []}
        for label in range(num_class)
    }
    for batch_idx, test_data in enumerate(tqdm(test_loader)):
        x, y = test_data[0], test_data[1]
        if gpu:
            x, y = x.cuda(), y.cuda()
        with torch.no_grad():
            y_pred = model(x)
        labels_cpu = y.data.cpu().float()
        pred_cpu = y_pred.data.cpu()
        all_metrics = metrics(labels_cpu.numpy(), pred_cpu.numpy(), train_stage=False)
        macro_metrics = all_metrics["macro avg"]
        precision.append(macro_metrics["precision"])
        recall.append(macro_metrics["recall"])
        f1.append(macro_metrics["f1-score"])
        for label_idx in range(num_class):
            key = str(label_idx)
            label_metrics[key]["precision"].append(all_metrics[key]["precision"])
            label_metrics[key]["recall"].append(all_metrics[key]["recall"])
            label_metrics[key]["f1"].append(all_metrics[key]["f1-score"])
    epoch_p = np.mean(precision)
    epoch_r = np.mean(recall)
    epoch_f = np.mean(f1)
    print()
    print("Test End")
    print()
    print("precision: %.4f, recall: %.4f, f1-score: %.4f" % (epoch_p, epoch_r, epoch_f))
    result_dict = dict()
    result_dict["macro_p"] = epoch_p
    result_dict["macro_r"] = epoch_r
    result_dict["macro_f1"] = epoch_f
    for label in range(num_class):
        key = str(label)
        result_dict[key] = {
            "precision": np.mean(label_metrics[key]["precision"]),
            "recall": np.mean(label_metrics[key]["recall"]),
            "f1": np.mean(label_metrics[key]["f1"])
        }
    with open(args.predict_result_path, "w", encoding="utf-8") as fp:
        json.dump(result_dict, fp, ensure_ascii=False, indent=4)


def metrics(true_mat, score_mat, threshold=0.5, train_stage=True):
    """
    Calculate macro precision, recall and f1 score
    """
    pred_mat = (score_mat >= threshold) * 1
    all_metrics = classification_report(true_mat, pred_mat, output_dict=True)
    macro_metrics = all_metrics["macro avg"]
    macro_precision = macro_metrics["precision"]
    macro_recall = macro_metrics["recall"]
    macro_f1 = macro_metrics["f1-score"]
    if train_stage:
        return macro_precision, macro_recall, macro_f1
    else:
        return all_metrics
