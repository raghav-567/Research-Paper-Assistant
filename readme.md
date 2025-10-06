Agentic-Research-Assistant/
|
│── src/
│   │── agents/          
│   │   ├── search_agent.py       # Queries Arxiv/Semantic Scholar
│   │   ├── extraction_agent.py   # PDF parsing + chunking
│   │   ├── summarizer_agent.py   # RAG summaries per paper
│   │   └── synthesizer_agent.py  # Aggregates into literature review
│   │
│   ├── rag_pipeline.py   # Embeddings + FAISS + Retriever
│   ├── utils.py          # Helper functions (logging, configs, formatting)
│   └── main.py           # Orchestrator: ties all agents together
│
│── app/
│   ├── streamlit_app.py  # Frontend demo
│   └── fastapi_app.py    # API backend (optional)
│
│── outputs/
│   ├── sample_review.md  # Generated structured lit review
│   └── references.bib
│
│── requirements.txt
│── README.md
