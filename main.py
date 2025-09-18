import os
import zipfile
import requests
import pandas as pd
from sklearn.model_selection import train_test_split
import torch

# Import project modules
from src.preprocess import clean_texts
from src.lstm_model import create_lstm_model, train_lstm, evaluate_lstm
from src.bert_model import create_bert_datasets, load_bert_model
from src.train import train_with_trainer, plot_confusion_matrix, print_classification_report

# -------------------------------------------------------------------
# 1. Download dataset if not exists
# -------------------------------------------------------------------
DATA_DIR = "data"
ZIP_PATH = os.path.join(DATA_DIR, "training.1600000.processed.noemoticon.csv.zip")
CSV_PATH = os.path.join(DATA_DIR, "training.1600000.processed.noemoticon.csv")

os.makedirs(DATA_DIR, exist_ok=True)

if not os.path.exists(CSV_PATH):
    print("Downloading dataset...")
    url = "https://nyc3.digitaloceanspaces.com/ml-files-distro/v1/sentiment-analysis-is-bad/data/training.1600000.processed.noemoticon.csv.zip"
    response = requests.get(url, stream=True)
    with open(ZIP_PATH, "wb") as f:
        f.write(response.content)

    print("Extracting dataset...")
    with zipfile.ZipFile(ZIP_PATH, "r") as zip_ref:
        zip_ref.extractall(DATA_DIR)

# -------------------------------------------------------------------
# 2. Load dataset
# -------------------------------------------------------------------
print("Loading dataset...")
df = pd.read_csv(CSV_PATH, encoding="latin-1", header=None)
df = df[[0, 5]]  # sentiment, text
df.columns = ["target", "text"]

# Convert sentiment {0: negative, 4: positive} → {0, 1}
df["target"] = df["target"].apply(lambda x: 0 if x == 0 else 1)

# -------------------------------------------------------------------
# 3. Preprocess text
# -------------------------------------------------------------------
print("Cleaning texts...")
df["clean_text"] = clean_texts(df["text"].tolist())

# Split data
train_texts, val_texts, train_labels, val_labels = train_test_split(
    df["clean_text"].tolist(),
    df["target"].tolist(),
    test_size=0.2,
    random_state=42
)

# -------------------------------------------------------------------
# 4. Train LSTM model
# -------------------------------------------------------------------
print("===== Training LSTM Model =====")
vocab_size = 20000
max_len = 100

lstm_model, tokenizer = create_lstm_model(vocab_size=vocab_size, max_len=max_len)
history_lstm = train_lstm(lstm_model, tokenizer, train_texts, train_labels, val_texts, val_labels)

loss, acc = evaluate_lstm(lstm_model, tokenizer, val_texts, val_labels)
print(f"LSTM Validation Accuracy: {acc:.4f}")

# -------------------------------------------------------------------
# 5. Train DistilBERT model
# -------------------------------------------------------------------
print("\n===== Training DistilBERT Model =====")
train_dataset, val_dataset, tokenizer = create_bert_datasets(train_texts, train_labels, val_texts, val_labels)
bert_model = load_bert_model(num_labels=2)

trainer = train_with_trainer(bert_model, tokenizer, train_dataset, val_dataset, epochs=3, batch_size=16)

# Evaluate BERT
results = trainer.evaluate()
print(f"DistilBERT Validation Accuracy: {results['eval_accuracy']:.4f}")

# -------------------------------------------------------------------
# 6. Compare Models
# -------------------------------------------------------------------
import matplotlib.pyplot as plt

lstm_acc = max(history_lstm.history['val_accuracy'])
bert_acc = results['eval_accuracy']

plt.bar(['LSTM', 'DistilBERT'], [lstm_acc, bert_acc], color=['blue', 'green'])
plt.title("Validation Accuracy Comparison")
plt.show()

# -------------------------------------------------------------------
# 7. Confusion Matrix + Report for DistilBERT
# -------------------------------------------------------------------
predictions = trainer.predict(val_dataset)
y_pred = predictions.predictions.argmax(axis=1)
y_true = predictions.label_ids

plot_confusion_matrix(y_true, y_pred, title="DistilBERT Confusion Matrix")
print_classification_report(y_true, y_pred, target_names=["Negative", "Positive"])