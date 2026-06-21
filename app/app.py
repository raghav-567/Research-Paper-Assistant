# app.py (Backend Server)
from flask import Flask, request, jsonify, render_template, session
from flask_cors import CORS
from dotenv import load_dotenv
import sys
import threading
import uuid
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from datetime import datetime
import os


sys.path.append(str(Path(__file__).parent.parent))

from src.main import generate_review
from src.utils import get_logger, get_api_key, mask_api_key

# Input bounds for the generate endpoint.
MAX_QUERY_LENGTH = 300
MIN_RESULTS = 1
MAX_RESULTS = 10
DEFAULT_RESULTS = 5

app = Flask(__name__)
logger = get_logger("FlaskApp")

load_dotenv()

# Prefer server-side sessions so the user's Gemini API key never travels to the
# browser. Flask's default session is a signed (not encrypted) cookie, which
# would expose the stored key to anyone who can read the cookie. Fall back to
# cookie sessions with a warning when Flask-Session isn't installed.
try:
    from flask_session import Session

    app.config["SESSION_TYPE"] = "filesystem"
    app.config["SESSION_FILE_DIR"] = os.getenv("SESSION_FILE_DIR", ".flask_session")
    app.config["SESSION_PERMANENT"] = False
    Session(app)
    logger.info("Server-side sessions enabled via Flask-Session")
except Exception as e:  # pragma: no cover - depends on optional dependency
    logger.warning(
        "Flask-Session unavailable; falling back to signed-cookie sessions "
        f"({e}). The Gemini API key would then be stored client-side — install "
        "Flask-Session before deploying."
    )

# Restrict CORS to known origins instead of allowing every site. Sessions hold
# the user's API key, so credentialed cross-origin requests must be scoped.
_DEFAULT_ORIGINS = "http://localhost:5001,http://127.0.0.1:5001"
_allowed_origins = [
    o.strip()
    for o in os.getenv("ALLOWED_ORIGINS", _DEFAULT_ORIGINS).split(",")
    if o.strip()
]
CORS(app, resources={r"/*": {"origins": _allowed_origins}}, supports_credentials=True)

# Session signing key. The default is insecure and only suitable for local dev.
_DEV_SECRET = "dev-only-change-me"
app.secret_key = os.getenv("FLASK_SECRET_KEY") or _DEV_SECRET
if app.secret_key == _DEV_SECRET:
    logger.warning(
        "FLASK_SECRET_KEY is not set; using an insecure default. "
        "Set FLASK_SECRET_KEY before deploying."
    )

# Optional per-client rate limiting. Falls back to a no-op when flask-limiter
# isn't installed so the app still runs in minimal environments.
try:
    from flask_limiter import Limiter
    from flask_limiter.util import get_remote_address

    limiter = Limiter(get_remote_address, app=app, default_limits=[])
    logger.info("Rate limiting enabled via flask-limiter")
except Exception as e:  # pragma: no cover - depends on optional dependency
    limiter = None
    logger.warning(f"flask-limiter unavailable; rate limiting disabled ({e})")


def _rate_limit(limit_value):
    """Apply a flask-limiter limit when available, otherwise a no-op."""
    def decorator(func):
        if limiter is not None:
            return limiter.limit(limit_value)(func)
        return func
    return decorator


# ---------------------------------------------------------------------------
# Background job system
#
# Review generation (arXiv search + PDF downloads + several LLM calls) takes
# tens of seconds to minutes, which is far longer than typical HTTP/proxy
# timeouts and gives the user no progress feedback. Instead of blocking the
# request, we run each review in a worker thread and let the client poll for
# status and progress.
# ---------------------------------------------------------------------------
_MAX_TRACKED_JOBS = 100
_jobs = OrderedDict()
_jobs_lock = threading.Lock()
_executor = ThreadPoolExecutor(max_workers=int(os.getenv("REVIEW_WORKERS", "2")))


def _create_job(query):
    job_id = uuid.uuid4().hex
    job = {
        "id": job_id,
        "status": "queued",
        "query": query,
        "progress": {"label": "Queued...", "value": 5},
        "result": None,
        "error": None,
        "created_at": datetime.now().isoformat(),
    }
    with _jobs_lock:
        _jobs[job_id] = job
        # Evict the oldest finished jobs once we exceed the cap.
        while len(_jobs) > _MAX_TRACKED_JOBS:
            _jobs.popitem(last=False)
    return job


def _update_job(job_id, **fields):
    with _jobs_lock:
        job = _jobs.get(job_id)
        if job is not None:
            job.update(fields)


def _get_job(job_id):
    with _jobs_lock:
        job = _jobs.get(job_id)
        return dict(job) if job is not None else None


def _run_review_job(job_id, query, output_file, api_key, max_results):
    def report(label, value):
        _update_job(job_id, progress={"label": label, "value": value})

    _update_job(job_id, status="running")
    report("Searching arXiv...", 15)
    try:
        result = generate_review(
            query=query,
            max_results=max_results,
            output_file=output_file,
            inter_paper_delay=0.5,
            api_key=api_key,
            progress_callback=report,
        )
        payload = {
            "review": result["review"],
            "papers_count": len(result["papers"]),
            "output_file": result["output_file"],
            "used_fallback_summary": result["used_fallback_summary"],
            "quota_error": result["quota_error"],
        }
        _update_job(
            job_id,
            status="done",
            result=payload,
            progress={"label": "Completed", "value": 100},
        )
    except (ValueError, RuntimeError) as e:
        logger.warning(f"Review job {job_id} failed: {e}")
        _update_job(job_id, status="error", error=str(e))
    except Exception as e:
        logger.exception(f"Unexpected error in review job {job_id}")
        _update_job(job_id, status="error", error=str(e))


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


@app.route('/seed-data.js')
def seed_data():
    # The template references seed-data.js, which only exists in the static
    # (Vercel) demo build. Serve an empty script here so the live Flask app
    # stays in normal backend mode without a 404 in the console.
    return app.response_class("", mimetype="application/javascript")


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

def _validated_max_results(data):
    """Clamp a client-supplied max_results into the allowed range."""
    raw = data.get("max_results", DEFAULT_RESULTS)
    try:
        value = int(raw)
    except (TypeError, ValueError):
        value = DEFAULT_RESULTS
    return max(MIN_RESULTS, min(MAX_RESULTS, value))


@app.route('/generate-review', methods=['POST'])
@_rate_limit("10 per minute")
def generate_literature_review():
    """Start a review generation job and return its id for polling."""
    data = request.get_json(silent=True) or {}
    query = data.get('query', '').strip()

    if not query:
        return jsonify({'error': 'Query is required'}), 400
    if len(query) > MAX_QUERY_LENGTH:
        return jsonify({
            'error': f'Query is too long (max {MAX_QUERY_LENGTH} characters).'
        }), 400

    runtime_api_key = _resolve_runtime_api_key(data)
    if not runtime_api_key:
        return jsonify({
            'error': 'Missing API key. Add your Gemini API key from the sidebar before generating a review.'
        }), 400

    max_results = _validated_max_results(data)
    request_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = f"outputs/review_{request_timestamp}.md"

    job = _create_job(query)
    # Resolve the API key here (request context) and hand it to the worker; the
    # session is not available off the request thread.
    _executor.submit(
        _run_review_job, job["id"], query, output_file, runtime_api_key, max_results
    )

    return jsonify({
        'job_id': job['id'],
        'status': job['status'],
        'timestamp': request_timestamp,
    }), 202


@app.route('/generate-review/<job_id>', methods=['GET'])
def review_job_status(job_id):
    """Return the status/progress of a review job, plus the result when done."""
    job = _get_job(job_id)
    if job is None:
        return jsonify({'error': 'Unknown job id.'}), 404

    response = {
        'job_id': job['id'],
        'status': job['status'],
        'progress': job['progress'],
    }
    if job['status'] == 'done' and job['result']:
        response.update(job['result'])
    elif job['status'] == 'error':
        response['error'] = job['error'] or 'Review generation failed.'

    return jsonify(response)


if __name__ == '__main__':
    # Never enable the Werkzeug debugger (an RCE console) unless explicitly
    # requested, and bind to localhost by default rather than all interfaces.
    debug = os.getenv("FLASK_DEBUG", "0").lower() in {"1", "true", "yes"}
    host = os.getenv("FLASK_HOST", "127.0.0.1")
    port = int(os.getenv("FLASK_PORT", "5001"))
    app.run(host=host, port=port, debug=debug)
