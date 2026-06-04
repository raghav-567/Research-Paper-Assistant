# app.py (Backend Server)
from flask import Flask, request, jsonify, render_template, session
from flask_cors import CORS
from dotenv import load_dotenv
import sys
from pathlib import Path
from datetime import datetime
import os


sys.path.append(str(Path(__file__).parent.parent))

from src.main import generate_review
from src.utils import get_logger, get_api_key, mask_api_key

app = Flask(__name__)
CORS(app)
logger = get_logger("FlaskApp")


load_dotenv()
app.secret_key = os.getenv("FLASK_SECRET_KEY", "dev-only-change-me")


def _normalize_api_key(value):
    key = (value or "").strip()
    return key if key else None


def _is_valid_gemini_key(key):
    if not key:
        return False
    key = key.strip()
    return len(key) >= 20 and " " not in key


def _resolve_runtime_api_key(payload=None):
    payload = payload or {}
    direct_key = _normalize_api_key(payload.get("api_key"))
    if direct_key:
        return direct_key

    session_key = _normalize_api_key(session.get("user_gemini_api_key"))
    if session_key:
        return session_key

    return get_api_key()

@app.route('/')
def home():
    return render_template('llm.html')

@app.route('/api-key/status', methods=['GET'])
def api_key_status():
    key = _resolve_runtime_api_key()
    source = "none"
    session_key = _normalize_api_key(session.get("user_gemini_api_key"))
    env_key = get_api_key()
    if session_key:
        source = "session"
    elif env_key:
        source = "environment"

    return jsonify({
        "configured": bool(key),
        "source": source,
        "masked_key": mask_api_key(key) if key else "",
    })


@app.route('/api-key', methods=['POST'])
def set_api_key():
    data = request.get_json(silent=True) or {}
    key = _normalize_api_key(data.get("api_key"))

    if not _is_valid_gemini_key(key):
        return jsonify({
            "error": "Please provide a valid Gemini API key.",
        }), 400

    session["user_gemini_api_key"] = key
    return jsonify({
        "configured": True,
        "source": "session",
        "masked_key": mask_api_key(key),
        "message": "Gemini API key saved for this browser session.",
    })


@app.route('/api-key', methods=['DELETE'])
def clear_api_key():
    session.pop("user_gemini_api_key", None)
    env_key = get_api_key()
    active_key = env_key if env_key else None
    return jsonify({
        "configured": bool(active_key),
        "source": "environment" if env_key else "none",
        "masked_key": mask_api_key(active_key) if active_key else "",
        "message": "Session API key cleared.",
    })

@app.route('/generate-review', methods=['POST'])
def generate_literature_review():
    try:
        data = request.get_json(silent=True) or {}
        query = data.get('query', '').strip()

        if not query:
            return jsonify({'error': 'Query is required'}), 400

        request_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = f"outputs/review_{request_timestamp}.md"
        runtime_api_key = _resolve_runtime_api_key(data)
        if not runtime_api_key:
            return jsonify({
                'error': 'Missing API key. Add your Gemini API key from the sidebar before generating a review.'
            }), 400

        result = generate_review(
            query=query,
            max_results=5,
            output_file=output_file,
            inter_paper_delay=0.5,
            api_key=runtime_api_key,
        )

        return jsonify({
            'review': result['review'],
            'papers_count': len(result['papers']),
            'timestamp': request_timestamp,
            'output_file': result['output_file'],
            'used_fallback_summary': result['used_fallback_summary'],
            'quota_error': result['quota_error'],
        })

    except ValueError as e:
        status_code = 400 if str(e) == 'Query is required.' else 500
        logger.warning(f"Request failed: {e}")
        return jsonify({'error': str(e)}), status_code
    except RuntimeError as e:
        logger.warning(f"Upstream service issue: {e}")
        return jsonify({'error': str(e)}), 503
    except Exception as e:
        logger.exception("Unexpected error while generating literature review")
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5001, debug=True)
