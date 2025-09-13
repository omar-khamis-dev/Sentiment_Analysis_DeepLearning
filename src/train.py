import sys
from src.data_loader import load_data
from src.preprocess import preprocess_texts
from src.lstm_model import build_lstm_model, prepare_lstm_data
from src.bert_model import build_bert_model, get_tokenizer
from sklearn.model_selection import train_test_split

def train(model_type="lstm"):
    df = load_data()
    
    if model_type == "lstm":
        X, y, tokenizer = preprocess_texts(df, method="lstm")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        
        model = build_lstm_model()
        model.fit(X_train, y_train, validation_split=0.2, epochs=2, batch_size=512)
        loss, acc = model.evaluate(X_test, y_test)
        print(f"LSTM Accuracy: {acc:.4f}")

    elif model_type == "bert":
        tokenizer = get_tokenizer()
        encodings = tokenizer(list(df["clean_text"]), truncation=True, padding=True, max_length=128)
        import tensorflow as tf
        X = dict(encodings)
        y = df["target"].values
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        
        model = build_bert_model()
        model.fit(X_train, y_train, validation_split=0.1, epochs=2, batch_size=16)
        loss, acc = model.evaluate(X_test, y_test)
        print(f"BERT Accuracy: {acc:.4f}")

if __name__ == "__main__":
    model_type = sys.argv[1] if len(sys.argv) > 1 else "lstm"
    train(model_type)
