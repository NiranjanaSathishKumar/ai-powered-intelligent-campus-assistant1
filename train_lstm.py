import json
import numpy as np
import pickle
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder

# Load intents
with open("intents.json") as f:
    data = json.load(f)

# Prepare training data
texts = []
labels = []
for intent in data["intents"]:
    for pattern in intent["patterns"]:
        texts.append(pattern)
        labels.append(intent["tag"])

# Tokenization
tokenizer = Tokenizer(num_words=2000, oov_token="<OOV>")
tokenizer.fit_on_texts(texts)
word_index = tokenizer.word_index
sequences = tokenizer.texts_to_sequences(texts)
padded = pad_sequences(sequences, maxlen=20, truncating='post')

# Label encoding
lbl_encoder = LabelEncoder()
labels_encoded = lbl_encoder.fit_transform(labels)

# Model definition
model = Sequential()
model.add(Embedding(2000, 64, input_length=20))
model.add(LSTM(64, return_sequences=False))
model.add(Dropout(0.3))
model.add(Dense(32, activation="relu"))
model.add(Dense(len(set(labels_encoded)), activation="softmax"))

model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

# Train
model.fit(padded, np.array(labels_encoded), epochs=300, verbose=1)

# Save
model.save("chat_model_lstm.h5")
pickle.dump(tokenizer, open("tokenizer.pkl", "wb"))
pickle.dump(lbl_encoder, open("label_encoder.pkl", "wb"))

print("✅ Model trained and saved successfully!")
