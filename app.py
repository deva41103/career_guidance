"""
Career Guidance Flask API + Web Showcase
Main entry point
"""

from flask import Flask
from flask_cors import CORS
from routes.guidance import guidance_bp
from routes.web import web_bp

app = Flask(__name__)
CORS(app)  # Allow requests from mobile/web app

# API routes — keep untouched for future app integration
app.register_blueprint(guidance_bp, url_prefix="/api")

# Web showcase
app.register_blueprint(web_bp)


@app.route("/health", methods=["GET"])
def health():
    return {"status": "ok", "service": "Career Guidance API"}, 200


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)