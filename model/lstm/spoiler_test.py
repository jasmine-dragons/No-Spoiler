import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model

np.random.seed(0)

train = pd.read_csv('train.csv')
cv = pd.read_csv('cv.csv')
train = pd.concat([train, cv], ignore_index=True)

train_text = (train['book_title'].map(str) + ' ~~~ ' + train['sentence'].map(str)).to_numpy()
train_labels = train['sent_spoil'].to_numpy().astype(np.int32)

test_text = [
    "I was so sad when Harry died at the end.",
    "I'm surprised John became the king.",
    "The monster was the real treasure we found along the way!",
    "Yesterday's wordle was pause.",
    "I thought Dimitri was hot",
    "At the end, Jonathan fucking kills everyone.",
    "I couldn't get this fucking software to run on my computer.",
    "my pp no hard",
    "Prometheus was cast down from Olympus and tortured for eternity."
]

reviewMaxLen = 600

tokenizer = Tokenizer(num_words=8000)
tokenizer.fit_on_texts(train_text)

model = load_model('model')

for text in test_text:
    seq = tokenizer.texts_to_sequences([text])
    pad = pad_sequences(seq, maxlen=reviewMaxLen)
    prediction = model.predict(pad).item()
    print(text, ":", prediction)