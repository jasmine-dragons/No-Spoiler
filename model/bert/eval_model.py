import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import torch
from transformers import DistilBertModel, DistilBertTokenizer
from train import SpoilerClassifier, get_predictions, create_data_loader

if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    MAX_LEN = 60
    BATCH_SIZE = 16

    PRE_TRAINED_MODEL_NAME = 'distilbert-base-uncased'
    tokenizer = DistilBertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)

    df_train = pd.read_csv("train.csv")
    df_test = pd.read_csv("test.csv")
    df = pd.concat([df_train, df_test])
    test_labels = df['spoiler'].to_numpy().astype(np.int32)

    test_data_loader = create_data_loader(df, tokenizer, MAX_LEN, BATCH_SIZE)

    class_names = ['no_spoilers', 'has_spoilers']
    model = SpoilerClassifier(len(class_names))
    model = model.to(device)
    model.load_state_dict(torch.load("distilbert_goodreads.bin"))

    y_review_texts, y_pred, y_pred_probs, y_test = get_predictions(
        model,
        test_data_loader
    )

    fpr_keras, tpr_keras, thresholds_keras = roc_curve(test_labels, y_pred)
    auc_keras = auc(fpr_keras, tpr_keras)
    print("AUC:", auc_keras)

    cm = confusion_matrix(test_labels, y_pred)
    display = ConfusionMatrixDisplay(cm, display_labels=['no spoiler', 'has spoiler'])
    display.plot(cmap='RdPu')
    plt.title("Confusion matrix of DistilBERT model trained on TV Tropes")
    plt.show()