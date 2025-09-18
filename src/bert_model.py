from transformers import DistilBertTokenizerFast, TFDistilBertForSequenceClassification

def load_bert_model(model_name="distilbert-base-uncased", num_classes=2):
    """
    Load DistilBERT model and tokenizer for text classification.

    Args:
        model_name (str): Hugging Face model name.
        num_classes (int): Number of output classes.

    Returns:
        tuple: (tokenizer, model)
    """
    # Load DistilBERT tokenizer
    tokenizer = DistilBertTokenizerFast.from_pretrained(model_name)

    # Load DistilBERT model for sequence classification
    model = TFDistilBertForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_classes
    )

    return tokenizer, model