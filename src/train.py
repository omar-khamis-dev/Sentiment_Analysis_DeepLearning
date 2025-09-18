import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

from transformers import Trainer, TrainingArguments

def train_with_trainer(model, tokenizer, train_dataset, val_dataset, output_dir="./results", epochs=3, batch_size=16):
    """
    Train a Hugging Face model using Trainer API.

    Args:
        model: Hugging Face model (e.g., DistilBERT).
        tokenizer: Hugging Face tokenizer.
        train_dataset: Tokenized training dataset.
        val_dataset: Tokenized validation dataset.
        output_dir (str): Directory to save model results.
        epochs (int): Number of training epochs.
        batch_size (int): Training batch size.

    Returns:
        Trainer: Trained model with Trainer API.
    """

    def compute_metrics(eval_pred):
        """
        Compute evaluation metrics for Trainer.
        """
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        acc = accuracy_score(labels, predictions)
        return {"accuracy": acc}

    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=50,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    return trainer


def plot_confusion_matrix(y_true, y_pred, title="Confusion Matrix"):
    """
    Plot confusion matrix using seaborn heatmap.
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(title)
    plt.show()


def print_classification_report(y_true, y_pred, target_names=None):
    """
    Print a classification report including precision, recall, and F1-score.
    """
    report = classification_report(y_true, y_pred, target_names=target_names)
    print(report)