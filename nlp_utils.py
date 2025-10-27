import pickle, json, numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from config import Config

# Don't load immediately - use lazy loading
_model = None
_tokenizer = None
_lbl_encoder = None
_intents = None

def _load_assets():
    global _model, _tokenizer, _lbl_encoder, _intents
    if _model is None:
        _model = load_model("chat_model_lstm.h5")
        _tokenizer = pickle.load(open("tokenizer.pkl", "rb"))
        _lbl_encoder = pickle.load(open("label_encoder.pkl", "rb"))
        with open("intents.json", "r", encoding="utf-8") as f:
            _intents = json.load(f)

def predict_intent(text):
    _load_assets()
    seq = _tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=20, truncating='post')
    preds = _model.predict(padded)[0]
    idx = np.argmax(preds)
    confidence = float(preds[idx])
    tag = _lbl_encoder.inverse_transform([idx])[0]
    return tag, confidence

def get_response(tag):
    _load_assets()
    for intent in _intents["intents"]:
        if intent["tag"] == tag:
            return np.random.choice(intent["responses"])
    return "I'm not sure I understand that."
