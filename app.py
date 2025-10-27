from flask import Flask, render_template, request, jsonify, session, g
from flask_cors import CORS
from models import db, ChatLog
from config import Config
from nlp_utils import predict_intent, get_response
from datetime import datetime
from sqlalchemy import func
import json, uuid

app = Flask(__name__)
app.config.from_object(Config)
db.init_app(app)
CORS(app)

with app.app_context():
    db.create_all()

@app.before_request
def assign_session():
    if "session_id" not in session:
        session["session_id"] = str(uuid.uuid4())
    g.session_id = session["session_id"]

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/chatbot")
def chatbot():
    return render_template("chatbot.html")

@app.route("/admin")
def admin_page():
    return render_template("admin.html")

@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.json
    msg = data.get("message", "").strip().lower()
    if not msg:
        return jsonify({"error": "Empty message"}), 400

    tag, confidence = predict_intent(msg)
    if confidence < Config.FALLBACK_THRESHOLD:
        tag = "fallback"

    response = get_response(tag)
    log = ChatLog(
        session_id=g.session_id,
        user_message=msg,
        bot_response=response,
        intent=tag,
        confidence=float(confidence)
    )
    db.session.add(log)
    db.session.commit()

    return jsonify({
        "user_message": msg,
        "bot_response": response,
        "intent": tag,
        "confidence": round(confidence, 3)
    })

@app.route("/api/analytics")
def analytics():
    total = db.session.query(func.count(ChatLog.id)).scalar() or 0
    by_intent = dict(db.session.query(ChatLog.intent, func.count(ChatLog.id))
                     .group_by(ChatLog.intent).all() or [])
    by_day = {str(d): c for d, c in db.session.query(func.date(ChatLog.timestamp),
                     func.count(ChatLog.id)).group_by(func.date(ChatLog.timestamp)).all() or []}

    return jsonify({
        "total": total,
        "per_intent": by_intent,
        "per_day": by_day
    })

@app.route("/api/logs")
def logs():
    logs = ChatLog.query.order_by(ChatLog.timestamp.desc()).limit(50).all()
    data = [{
        "timestamp": log.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
        "user_message": log.user_message,
        "bot_response": log.bot_response,
        "intent": log.intent,
        "confidence": round(log.confidence, 2)
    } for log in logs]
    return jsonify(data)

if __name__ == "__main__":
    app.run(debug=True)
