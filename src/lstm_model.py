from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout

# Data processing
max_words = 20000
max_len = 100

tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(df["clean_text"])
sequences = tokenizer.texts_to_sequences(df["clean_text"])
X = pad_sequences(sequences, maxlen=max_len)
y = df["target"].values

# LSTM model
model = Sequential([
    Embedding(input_dim=max_words, output_dim=128),
    LSTM(128, dropout=0.2, recurrent_dropout=0.2),
    Dense(1, activation="sigmoid")
])

model.build(input_shape=(None, max_len))

model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
print(model.summary())