import string
from wordcloud import WordCloud
import pandas as pd
from collections import defaultdict
import re
import nltk
from nltk.corpus import stopwords
import matplotlib.pyplot as plt

train = pd.read_csv('tvtropes/train.balanced.csv')
test = pd.read_csv('tvtropes/test.balanced.csv')

all = pd.concat([train, test])

all_spoiler = all[all['spoiler'] == True]

word_freq = defaultdict(int)

punc = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''
stops = stopwords.words('english')

for sentence in all_spoiler['sentence']:
    sentence = re.sub(r'[^\w\s]', ' ', sentence)
    for word in sentence.split():
        word = word.lower()
        if word in stops:
            continue
        word_freq[word] += 1

cloud = WordCloud(font_path="ArcaneNine.otf", width=1920, height=1080, colormap="plasma")
cloud.generate_from_frequencies(word_freq)

plt.imshow(cloud, interpolation='bilinear')
plt.axis("off")
plt.show()