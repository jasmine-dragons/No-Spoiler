# Commented out IPython magic to ensure Python compatibility.
import numpy as np
import pandas as pd
import random
import transformers
from transformers import BertModel, BertTokenizer, AdamW, get_linear_schedule_with_warmup
from transformers import DistilBertModel, DistilBertTokenizer
import torch
import seaborn as sns
from pylab import rcParams
import matplotlib.pyplot as plt
from matplotlib import rc
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import roc_auc_score
from collections import defaultdict
from textwrap import wrap
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import tqdm as tq

class GPReviewDataset(Dataset):
  def __init__(self, reviews, targets, tokenizer, max_len):
    self.reviews = reviews
    self.targets = targets
    self.tokenizer = tokenizer
    self.max_len = max_len
  def __len__(self):
    return len(self.reviews)
  def __getitem__(self, item):
    review = str(self.reviews[item])
    target = self.targets[item]
    encoding = self.tokenizer.encode_plus(
      review,
      add_special_tokens=True,
      max_length=self.max_len,
      return_token_type_ids=False,
      pad_to_max_length=True,
      return_attention_mask=True,
      return_tensors='pt',
      truncation=True,
    )
    return {
      'review_text': review,
      'input_ids': encoding['input_ids'].flatten(),
      'attention_mask': encoding['attention_mask'].flatten(),
      'targets': torch.tensor(target, dtype=torch.long)
    }

def create_data_loader(df, tokenizer, max_len, batch_size):
  ds = GPReviewDataset(
    reviews=(df.page + ' ~~~ ' + df.sentence).to_numpy(),
    targets=df.spoiler.to_numpy(),
    tokenizer=tokenizer,
    max_len=max_len
  )
  return DataLoader(
    ds,
    batch_size=batch_size,
    num_workers=4
  )

class SpoilerClassifier(nn.Module):
  def __init__(self, n_classes):
    super(SpoilerClassifier, self).__init__()
    PRE_TRAINED_MODEL_NAME = 'distilbert-base-uncased'
    self.bert = DistilBertModel.from_pretrained(PRE_TRAINED_MODEL_NAME)
    # self.bert = BertModel.from_pretrained(PRE_TRAINED_MODEL_NAME)
    self.drop = nn.Dropout(p=0.3)
    self.out = nn.Linear(self.bert.config.hidden_size, n_classes)
  def forward(self, input_ids, attention_mask):
    model_output = self.bert(
      input_ids=input_ids,
      attention_mask=attention_mask
    )
    last_hidden_state = model_output[0]
    pooled_output = last_hidden_state[:, 0, :]
    output = self.drop(pooled_output)
    return self.out(output)

def train_epoch(
  model,
  data_loader,
  loss_fn,
  optimizer,
  device,
  scheduler,
  n_examples
):
  model = model.train()
  losses = []
  correct_predictions = 0
  for d in data_loader:
    input_ids = d["input_ids"].to(device)
    attention_mask = d["attention_mask"].to(device)
    targets = d["targets"].to(device)
    outputs = model(
      input_ids=input_ids,
      attention_mask=attention_mask
    )
    _, preds = torch.max(outputs, dim=1)
    loss = loss_fn(outputs, targets)
    correct_predictions += torch.sum(preds == targets)
    losses.append(loss.item())
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad()
  return correct_predictions.double() / n_examples, np.mean(losses)

def eval_model(model, data_loader, loss_fn, device, n_examples):
  model = model.eval()
  losses = []
  correct_predictions = 0
  with torch.no_grad():
    for d in data_loader:
      input_ids = d["input_ids"].to(device)
      attention_mask = d["attention_mask"].to(device)
      targets = d["targets"].to(device)
      outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask
      )
      _, preds = torch.max(outputs, dim=1)
      loss = loss_fn(outputs, targets)
      correct_predictions += torch.sum(preds == targets)
      losses.append(loss.item())
  return correct_predictions.double() / n_examples, np.mean(losses)

def get_predictions(model, data_loader):
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  model = model.eval()
  review_texts = []
  predictions = []
  prediction_probs = []
  real_values = []
  with torch.no_grad():
    for d in tq.tqdm(data_loader):
      texts = d["review_text"]
      input_ids = d["input_ids"].to(device)
      attention_mask = d["attention_mask"].to(device)
      targets = d["targets"].to(device)
      outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask
      )
      _, preds = torch.max(outputs, dim=1)
      review_texts.extend(texts)
      predictions.extend(preds)
      prediction_probs.extend(outputs)
      real_values.extend(targets)
  predictions = torch.stack(predictions).cpu()
  prediction_probs = torch.stack(prediction_probs).cpu()
  real_values = torch.stack(real_values).cpu()
  return review_texts, predictions, prediction_probs, real_values

def show_confusion_matrix(confusion_matrix):
  hmap = sns.heatmap(confusion_matrix, annot=True, fmt="d", cmap="Blues")
  hmap.yaxis.set_ticklabels(hmap.yaxis.get_ticklabels(), rotation=0, ha='right')
  hmap.xaxis.set_ticklabels(hmap.xaxis.get_ticklabels(), rotation=30, ha='right')
  plt.ylabel('True sentiment')
  plt.xlabel('Predicted sentiment')

if __name__ == '__main__':
  sns.set(style='whitegrid', palette='muted', font_scale=1.2)
  HAPPY_COLORS_PALETTE = ["#01BEFE", "#FFDD00", "#FF7D00", "#FF006D", "#ADFF02", "#8F00FF"]
  sns.set_palette(sns.color_palette(HAPPY_COLORS_PALETTE))
  rcParams['figure.figsize'] = 12, 8
  RANDOM_SEED = 42
  np.random.seed(RANDOM_SEED)
  torch.manual_seed(RANDOM_SEED)
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

  df = pd.read_csv("train.csv")
  class_names = ['no_spoilers', 'has_spoilers']
  ax = sns.countplot(df.spoiler)
  plt.xlabel('review spoiler')
  ax.set_xticklabels(class_names)

  PRE_TRAINED_MODEL_NAME = 'distilbert-base-uncased'
  # PRE_TRAINED_MODEL_NAME = 'bert-base-cased'

  tokenizer = DistilBertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)
  # tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)

  token_lens = []
  for txt in tq.tqdm(df.sentence):
    tokens = tokenizer.encode(txt, max_length=128, truncation=True)
    token_lens.append(len(tokens))

  sns.displot(token_lens)
  plt.xlim([0, 256])
  plt.xlabel('Token count')

  MAX_LEN = 60

  df_train, df_test = train_test_split(
    df,
    test_size=0.2,
    random_state=RANDOM_SEED
  )
  df_val, df_test = train_test_split(
    df_test,
    test_size=0.5,
    random_state=RANDOM_SEED
  )

  df_train.shape, df_val.shape, df_test.shape

  BATCH_SIZE = 16

  train_data_loader = create_data_loader(df_train, tokenizer, MAX_LEN, BATCH_SIZE)
  val_data_loader = create_data_loader(df_val, tokenizer, MAX_LEN, BATCH_SIZE)
  test_data_loader = create_data_loader(df_test, tokenizer, MAX_LEN, BATCH_SIZE)

  data = next(iter(train_data_loader))
  data.keys()

  print(data['input_ids'].shape)
  print(data['attention_mask'].shape)
  print(data['targets'].shape)

  bert_model = DistilBertModel.from_pretrained(PRE_TRAINED_MODEL_NAME)
  # bert_model = BertModel.from_pretrained(PRE_TRAINED_MODEL_NAME)

  model = SpoilerClassifier(len(class_names))
  model = model.to(device)

  input_ids = data['input_ids'].to(device)
  attention_mask = data['attention_mask'].to(device)
  print(input_ids.shape) # batch size x seq length
  print(attention_mask.shape) # batch size x seq length

  nn.functional.softmax(model(input_ids, attention_mask), dim=1)

  EPOCHS = 3
  optimizer = AdamW(model.parameters(), lr=2e-5, correct_bias=False)
  total_steps = len(train_data_loader) * EPOCHS
  scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=0,
    num_training_steps=total_steps
  )
  loss_fn = nn.CrossEntropyLoss().to(device)

  history = defaultdict(list)
  best_accuracy = 0
  for epoch in range(EPOCHS):
    print(f'Epoch {epoch + 1}/{EPOCHS}')
    print('-' * 10)
    train_acc, train_loss = train_epoch(
      model,
      train_data_loader,
      loss_fn,
      optimizer,
      device,
      scheduler,
      len(df_train)
    )
    print(f'Train loss {train_loss} accuracy {train_acc}')
    val_acc, val_loss = eval_model(
      model,
      val_data_loader,
      loss_fn,
      device,
      len(df_val)
    )
    print(f'Val   loss {val_loss} accuracy {val_acc}')
    print()
    history['train_acc'].append(train_acc)
    history['train_loss'].append(train_loss)
    history['val_acc'].append(val_acc)
    history['val_loss'].append(val_loss)
    if val_acc > best_accuracy:
      torch.save(model.state_dict(), 'best_model_state.bin')
      best_accuracy = val_acc

  plt.plot(history['train_acc'], label='train accuracy')
  plt.plot(history['val_acc'], label='validation accuracy')
  plt.title('Training history')
  plt.ylabel('Accuracy')
  plt.xlabel('Epoch')
  plt.legend()
  plt.ylim([0, 1])

  test_acc, _ = eval_model(
    model,
    test_data_loader,
    loss_fn,
    device,
    len(df_test)
  )
  test_acc.item()

  y_review_texts, y_pred, y_pred_probs, y_test = get_predictions(
    model,
    test_data_loader
  )

  print(classification_report(y_test, y_pred, target_names=class_names))

  print(roc_auc_score(y_test, y_pred))

  cm = confusion_matrix(y_test, y_pred)
  df_cm = pd.DataFrame(cm, index=class_names, columns=class_names)
  show_confusion_matrix(df_cm)

  idx = 2
  review_text = y_review_texts[idx]
  true_sentiment = y_test[idx]
  pred_df = pd.DataFrame({
    'class_names': class_names,
    'values': y_pred_probs[idx]
  })

  print("\n".join(wrap(review_text)))
  print()
  print(f'True sentiment: {class_names[true_sentiment]}')

  sns.barplot(x='values', y='class_names', data=pred_df, orient='h')
  plt.ylabel('sentiment')
  plt.xlabel('probability')
  plt.xlim([0, 1])

  sns.barplot(x='values', y='class_names', data=pred_df, orient='h')
  plt.ylabel('spoiler')
  plt.xlabel('probability')
  plt.xlim([0, 1])

