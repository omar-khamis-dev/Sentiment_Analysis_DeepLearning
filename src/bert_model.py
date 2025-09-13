from transformers import DistilBertTokenizerFast, TFDistilBertForSequenceClassification
from tensorflow.keras.optimizers import Adam

def build_bert_model(num_labels=2):
    # تحميل DistilBERT مُهيأ لتصنيف ثنائي
    model = TFDistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-uncased", 
        num_labels=num_labels
    )
    
    # إعداد optimizer
    optimizer = Adam(learning_rate=5e-5)
    model.compile(optimizer=optimizer, 
                  loss=model.compute_loss, 
                  metrics=["accuracy"])
    
    return model

def get_tokenizer():
    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
    return tokenizer
