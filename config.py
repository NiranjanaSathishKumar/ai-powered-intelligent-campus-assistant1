import os

class Config:
    SECRET_KEY = os.getenv("SECRET_KEY", "dev_key")
    SQLALCHEMY_DATABASE_URI = "sqlite:///chatbot.db"
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    FALLBACK_THRESHOLD = 0.6

