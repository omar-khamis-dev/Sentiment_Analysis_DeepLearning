import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout

def build_lstm_model(vocab_size, embedding_dim=128, lstm_units=128, dropout_rate=0.3, num_classes=2):
    """
    Build an LSTM model for text classification.

    Args:
        vocab_size (int): Size of the vocabulary.
        embedding_dim (int): Dimension of word embeddings.
        lstm_units (int): Number of LSTM units.
        dropout_rate (float): Dropout rate to prevent overfitting.
        num_classes (int): Number of output classes.

    Returns:
        tf.keras.Model: Compiled LSTM model.
    """
    model = Sequential()

    # Embedding layer to learn word representations
    model.add(Embedding(input_dim=vocab_size, output_dim=embedding_dim, mask_zero=True))

    # LSTM layer
    model.add(LSTM(units=lstm_units, return_sequences=False))

    # Dropout for regularization
    model.add(Dropout(dropout_rate))

    # Dense output layer with softmax for classification
    model.add(Dense(num_classes, activation="softmax"))

    # Compile the model
    model.compile(loss="sparse_categorical_crossentropy",
                  optimizer="adam",
                  metrics=["accuracy"])

    return model