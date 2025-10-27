import pickle, json, numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from config import Config

# Load assets
model = load_model("chat_model_lstm.h5")
tokenizer = pickle.load(open("tokenizer.pkl", "rb"))
lbl_encoder = pickle.load(open("label_encoder.pkl", "rb"))

with open("intents.json", "r", encoding="utf-8") as f:
    intents = json.load(f)



def predict_intent(text):
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=20, truncating='post')
    preds = model.predict(padded)[0]
    idx = np.argmax(preds)
    confidence = float(preds[idx])
    tag = lbl_encoder.inverse_transform([idx])[0]
    return tag, confidence


def get_response(tag):
    for intent in intents["intents"]:
        if intent["tag"] == tag:
            return np.random.choice(intent["responses"])
    return "I'm not sure I understand that."
