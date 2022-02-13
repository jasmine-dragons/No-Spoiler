import numpy as np
import pandas as pd
from tensorflow import keras
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

train = pd.read_csv("../data/tvtropes/train.balanced.csv")
test = pd.read_csv("../data/tvtropes/test.balanced.csv")

df = test

train_text = (train['page'].map(str) + ' ~~~ ' + train['sentence'].map(str)).to_numpy()

text = (df['page'].map(str) + ' ~~~ ' + df['sentence'].map(str)).to_numpy()
labels = df['spoiler'].to_numpy().astype(np.int32)

reviewMaxLen = 600

tokenizer = Tokenizer(num_words=8000)
tokenizer.fit_on_texts(train_text)
sequences = tokenizer.texts_to_sequences(train_text)
padded = pad_sequences(sequences, maxlen=reviewMaxLen)

model = load_model('model')

# Run model on test set to predict new reviews
predictions = []
i = 0
for t in text:
    if i % 100 == 0:
        print(i)
    i += 1
    seq = tokenizer.texts_to_sequences([t])
    pad = pad_sequences(seq, maxlen=reviewMaxLen)
    prediction = model.predict(pad).item()
    predictions.append(prediction)
predictions = np.array(predictions)

# Make predictions based on confidence
predictions = [1 if predict >= .50 else 0 for predict in predictions]

from sklearn.metrics import roc_curve
from sklearn.metrics import auc
fpr_keras, tpr_keras, thresholds_keras = roc_curve(labels, predictions)
auc_keras = auc(fpr_keras, tpr_keras)
print("AUC:", auc_keras)

cm = confusion_matrix(labels, predictions)
display = ConfusionMatrixDisplay(cm, display_labels=['no spoiler', 'has spoiler'])
display.plot(cmap='RdPu')
plt.title('Confusion matrix of LSTM model trained on Goodreads')
plt.show()