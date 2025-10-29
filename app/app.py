# app.py (Backend Server)
from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from dotenv import load_dotenv
import time
import faiss
import sys
from pathlib import Path
from datetime import datetime
from flask import Flask, request, jsonify, render_template


sys.path.append(str(Path(__file__).parent.parent))

from src.agents.search_agent import SearchAgent
from src.agents.extraction_agent import ExtractionAgent
from src.agents.summarizer_agent import SummarizerAgent
from src.agents.synthesizer_agent import SynthesizerAgent
from src.rag_pipeline import RAGPipeline

app = Flask(__name__)
CORS(app)


load_dotenv()

@app.route('/')
def home():
    return render_template('llm.html')

@app.route('/generate-review', methods=['POST'])
def generate_literature_review():
    try:
        data = request.get_json(force=True)
        query = data.get('query', '').strip()

        if not query:
            return jsonify({'error': 'Query is required'}), 400

        api_key = os.getenv("API_KEY")
        if not api_key:
            return jsonify({'error': 'API_KEY not found in environment'}), 500
        
        search = SearchAgent(pdf_dir="pdfs")
        extraction = ExtractionAgent()
        summarizer = SummarizerAgent(api_key=api_key)
        synthesizer = SynthesizerAgent()

        dim = summarizer.embedding_model.get_sentence_embedding_dimension()
        index = faiss.IndexFlatL2(dim)
        rag = RAGPipeline(search, extraction, summarizer, index, {})

        papers = search.search_arxiv(query, max_results=5)
        summaries = []

        for paper in papers:
            summary = summarizer._summarize_text(
                paper.get("summary", ""), max_output_tokens=300
            )
            summaries.append(summary)
            time.sleep(0.5)

        review = synthesizer.synthesize(papers, summaries)

        return jsonify({
            'review': review,
            'papers_count': len(papers),
            'timestamp': datetime.now().strftime('%Y%m%d_%H%M%S')
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
   app.run(host="0.0.0.0", port=5001, debug=True)

