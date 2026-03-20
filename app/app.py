# app.py (Backend Server)
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from dotenv import load_dotenv
import sys
from pathlib import Path
from datetime import datetime


sys.path.append(str(Path(__file__).parent.parent))

from src.main import generate_review
from src.utils import get_logger

app = Flask(__name__)
CORS(app)
logger = get_logger("FlaskApp")


load_dotenv()

@app.route('/')
def home():
    return render_template('llm.html')

@app.route('/generate-review', methods=['POST'])
def generate_literature_review():
    try:
        data = request.get_json(silent=True) or {}
        query = data.get('query', '').strip()

        if not query:
            return jsonify({'error': 'Query is required'}), 400

        request_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = f"outputs/review_{request_timestamp}.md"
        result = generate_review(
            query=query,
            max_results=5,
            output_file=output_file,
            inter_paper_delay=0.5,
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
